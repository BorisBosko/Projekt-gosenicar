import mujoco                           # Glavna knjiznica MuJoCo za model, podatke in simulacijo
import mujoco.viewer as viewer          # Viewer za 3D prikaz simulacije
import numpy as np                      # NumPy za racune z vektorji, koti, tabelami
import threading                        # Za loceno nit fizike
import time                             # Za merjenje casa in usklajevanje korakov simulacije
import tkinter as tk                    # Za enostaven GUI z gumbi
from pathlib import Path                # Za delo s potmi do datotek
from dataclasses import dataclass          # Za jasne parametre G4 terramechanics core
from PIL import Image, ImageTk
import csv
import matplotlib
matplotlib.use("Agg")  # shrani grafe v PNG tudi, ko tece MuJoCo viewer
import matplotlib.pyplot as plt


@dataclass
class TerramechanicsParams:
    """
    Parametri modula G4.

    Modul uporablja vhode:
      - vcmd, omega_cmd,
      - teren: mu, lambda,
      - obremenitve: Fz_left, Fz_right,
    in vrne:
      - FL, FR,
      - rolling resistance,
      - scrub moment,
      - debug podatke o zdrsu.
    """
    mass_kg: float = 35.0
    track_width_m: float = 1.0
    yaw_inertia_kgm2: float = 8.0

    # Hitrostni regulator, ki iz napake hitrosti naredi pogonsko silo/moment.
    # Enota je priblizno 1/s. Prejsnja verzija je uporabljala prevelike vrednosti,
    # zato je regulator skoraj ves cas udarjal v tanh saturacijo in povzročal
    # pocasno-hitro-pocasno nihanje pri pivotu.
    kv_force: float = 3.0
    kw_moment: float = 3.6

    # Najvecja pogonska sila na eno stran, preden jo se omeji oprijem mu*Fz.
    max_side_force_N: float = 520.0

    # Omejitev hitrosti spreminjanja sil/momentov. To je notranja dinamika G4,
    # zato se posodablja s fiksnim dt = model.opt.timestep.
    force_rate_limit_N_per_s: float = 6000.0
    moment_rate_limit_Nm_per_s: float = 4500.0

    # Kotalni upor: crr = crr_base + crr_lambda_gain*lambda + crr_mu_gain*(1-mu_norm)
    # Vrednosti so namenoma zmerne, da krog/osmica ne izgubita prevec hitrosti.
    crr_base: float = 0.015
    crr_lambda_gain: float = 0.045
    crr_mu_gain: float = 0.015

    # Scrub torque pri zavijanju na mestu / drsenju gosenic.
    # Scrub mora biti opazen, ne sme pa premagati pogona in povzrociti ciklicnega
    # ustavljanja pivot obrata.
    scrub_gain: float = 0.032
    scrub_omega_scale: float = 1.20
    scrub_compensation: float = 0.65

    # Meje zdrsa.
    sigma_v_min: float = 0.02
    sigma_v_max: float = 0.92
    sigma_omega_min: float = 0.03
    sigma_omega_max: float = 0.95

    # Filtriranje ukazov, da pri vecji hitrosti tank ne dobi sunkov.
    motor_tau_s: float = 0.08


class TerramechanicsCore:
    """
    G4 - terramechanics/slip core.

    To je locen modul z API-jem:
        reset()
        step(dt, cmd, terrain, loads, state) -> forces_and_torques
        get_state()

    V runtime zanki uporablja fiksni dt iz MuJoCo model.opt.timestep.
    """

    def __init__(self, params: TerramechanicsParams):
        self.p = params
        self._v_filt = 0.0
        self._omega_filt = 0.0
        self._F_total_cmd_prev = 0.0
        self._M_cmd_prev = 0.0
        self._last = {}

    @staticmethod
    def _clip(value, lo, hi):
        return float(max(lo, min(hi, value)))

    @staticmethod
    def _rate_limit(previous: float, target: float, max_rate: float, dt: float) -> float:
        """Omeji spremembo signala na max_rate na sekundo pri fiksnem dt."""
        previous = float(previous)
        target = float(target)
        step = max(0.0, float(max_rate)) * max(1e-6, float(dt))
        return previous + max(-step, min(step, target - previous))

    @staticmethod
    def _sign_nonzero(value, fallback=1.0):
        value = float(value)
        if abs(value) > 1e-6:
            return float(np.sign(value))
        fallback = float(fallback)
        if abs(fallback) > 1e-6:
            return float(np.sign(fallback))
        return 1.0

    def reset(self):
        self._v_filt = 0.0
        self._omega_filt = 0.0
        self._F_total_cmd_prev = 0.0
        self._M_cmd_prev = 0.0
        self._last = {}

    def slip_from_terrain(self, mu: float, lam: float, vcmd: float, omega_cmd: float):
        """
        Iz (mu, lambda) in ukazov oceni sigma_v in sigma_omega.

        Vecji lambda -> mehak teren -> vec zdrsa.
        Manjsi mu -> slabsi oprijem -> vec zdrsa.
        Vecji |omega_cmd| -> vec scrub/zdrsa pri zavijanju.
        """
        mu = float(mu)
        lam = self._clip(lam, 0.0, 1.0)

        # Normalizacija mu: pri mu >= 1.2 ga obravnavamo kot zelo dober oprijem.
        mu_norm = self._clip(mu / 1.2, 0.0, 1.0)

        # Osnovna zahteva G4: sigma_v(lambda) in sigma_omega(lambda).
        # Dodan je majhen popravek z mu in ukazom, toda dominantna odvisnost je lambda.
        sigma_v_lambda = self.p.sigma_v_min + 0.32 * lam
        sigma_omega_lambda = self.p.sigma_omega_min + 0.36 * lam

        sigma_v = (
            sigma_v_lambda
            + 0.07 * (1.0 - mu_norm)
            + 0.012 * min(abs(float(vcmd)), 2.5)
        )
        sigma_omega = (
            sigma_omega_lambda
            + 0.08 * (1.0 - mu_norm)
            + 0.012 * min(abs(float(omega_cmd)), 4.0)
        )

        sigma_v = self._clip(sigma_v, self.p.sigma_v_min, self.p.sigma_v_max)
        sigma_omega = self._clip(sigma_omega, self.p.sigma_omega_min, self.p.sigma_omega_max)
        return sigma_v, sigma_omega

    def step(self, dt: float, cmd: dict, terrain: dict, loads: dict, state: dict):
        """
        Glavni korak G4.

        Parametri:
          dt      - fiksni MuJoCo timestep,
          cmd     - {"v_cmd": ..., "omega_cmd": ...},
          terrain - {"mu": ..., "lambda": ...},
          loads   - {"Fz_left": ..., "Fz_right": ...},
          state   - {"v_forward": ..., "yaw_rate": ...}
        """
        dt = max(1e-6, float(dt))
        mu = float(terrain.get("mu", 0.8))
        lam = self._clip(float(terrain.get("lambda", 0.5)), 0.0, 1.0)

        vcmd_raw = float(cmd.get("v_cmd", 0.0))
        omega_raw = float(cmd.get("omega_cmd", 0.0))

        # Fiksno-casovno filtriranje ukaza. To je notranja dinamika, zato uporablja isti dt
        # kot MuJoCo, kot zahteva G4.
        alpha = self._clip(dt / max(self.p.motor_tau_s, dt), 0.0, 1.0)
        self._v_filt += alpha * (vcmd_raw - self._v_filt)
        self._omega_filt += alpha * (omega_raw - self._omega_filt)

        vcmd = self._v_filt
        omega_cmd = self._omega_filt

        Fz_left = max(0.0, float(loads.get("Fz_left", 0.0)))
        Fz_right = max(0.0, float(loads.get("Fz_right", 0.0)))
        Fz_total = Fz_left + Fz_right

        # Ce MuJoCo v prvem koraku se nima kontaktov, uporabimo oceno iz mase.
        if Fz_total < 1.0:
            Fz_total = max(1.0, self.p.mass_kg * 9.81)
            Fz_left = 0.5 * Fz_total
            Fz_right = 0.5 * Fz_total

        v_body = float(state.get("v_forward", 0.0))
        yaw_rate = float(state.get("yaw_rate", 0.0))

        sigma_v, sigma_omega = self.slip_from_terrain(mu, lam, vcmd, omega_cmd)
        v_ref = vcmd * (1.0 - sigma_v)
        omega_ref = omega_cmd * (1.0 - sigma_omega)

        # Hitrostna napaka -> sila/moment. tanh saturacija prepreči "letenje" pri sunkih.
        # Pomembno: regulator NI vec bang-bang. Zmerni gaini + rate limit odstranijo
        # ciklicno pospesevanje/zaviranje, ki se je videlo pri pivotu.
        F_total_request = self.p.kv_force * self.p.mass_kg * (v_ref - v_body)

        # Feed-forward kompenzacija scrub upora: ce zelimo stalen pivot, mora pogon
        # ze vnaprej premagati del striznega upora tal. Brez tega se pivot pri dosezeni
        # kotni hitrosti upocasni, potem regulator spet sunkovito pospesi.
        b_for_scrub = max(0.05, float(self.p.track_width_m))
        Fz_for_scrub = max(1.0, Fz_total)
        scrub_ff_mag = (
            self.p.scrub_gain
            * mu
            * (1.0 + 1.4 * lam)
            * Fz_for_scrub
            * 0.5 * b_for_scrub
            * np.tanh(abs(omega_ref) / max(self.p.scrub_omega_scale, 1e-6))
        )
        M_scrub_feedforward = self.p.scrub_compensation * self._sign_nonzero(omega_ref, fallback=omega_cmd) * scrub_ff_mag

        M_request = (
            self.p.kw_moment * self.p.yaw_inertia_kgm2 * (omega_ref - yaw_rate)
            + M_scrub_feedforward
        )

        # Globalna omejitev pogona po sili in momentu.
        F_total_limit = 2.0 * self.p.max_side_force_N
        M_limit = self.p.max_side_force_N * self.p.track_width_m
        F_total_target = F_total_limit * np.tanh(F_total_request / max(F_total_limit, 1e-6))
        M_target = M_limit * np.tanh(M_request / max(M_limit, 1e-6))

        # Fiksno-casovno omejevanje spremembe sil/momentov. To je notranja dinamika
        # G4 in uporablja natanko isti dt kot MuJoCo.
        F_total_cmd = self._rate_limit(
            self._F_total_cmd_prev,
            F_total_target,
            self.p.force_rate_limit_N_per_s,
            dt,
        )
        M_cmd = self._rate_limit(
            self._M_cmd_prev,
            M_target,
            self.p.moment_rate_limit_Nm_per_s,
            dt,
        )
        self._F_total_cmd_prev = float(F_total_cmd)
        self._M_cmd_prev = float(M_cmd)

        b = max(0.05, float(self.p.track_width_m))

        # Pretvorba vsota/moment -> leva/desna pogonska sila.
        FL_drive = 0.5 * F_total_cmd - M_cmd / b
        FR_drive = 0.5 * F_total_cmd + M_cmd / b

        # Omejitev s Coulombovim oprijemom po posamezni strani: |F| <= mu * Fz.
        # Na mehki podlagi dodatno pade ucinkovit oprijem.
        terrain_eff = self._clip(1.0 - 0.45 * lam, 0.25, 1.0)
        FL_cap = max(15.0, mu * terrain_eff * Fz_left)
        FR_cap = max(15.0, mu * terrain_eff * Fz_right)
        FL_drive = FL_cap * np.tanh(FL_drive / max(FL_cap, 1e-6))
        FR_drive = FR_cap * np.tanh(FR_drive / max(FR_cap, 1e-6))

        # Rolling resistance je sorazmeren z normalno silo in raste z lambda.
        mu_norm = self._clip(mu / 1.2, 0.0, 1.0)
        crr = self.p.crr_base + self.p.crr_lambda_gain * lam + self.p.crr_mu_gain * (1.0 - mu_norm)

        if abs(v_body) > 1e-4 or abs(vcmd) > 1e-4:
            rr_sign = self._sign_nonzero(v_body, fallback=vcmd)
            Frr_left = -rr_sign * crr * Fz_left
            Frr_right = -rr_sign * crr * Fz_right
        else:
            Frr_left = 0.0
            Frr_right = 0.0

        # Scrub torque upira relativnemu vrtenju na mestu.
        scrub_sign = self._sign_nonzero(yaw_rate, fallback=omega_cmd)
        Mscrub_mag = (
            self.p.scrub_gain
            * mu
            * (1.0 + 1.4 * lam)
            * Fz_total
            * 0.5 * b
            * np.tanh(abs(yaw_rate if abs(yaw_rate) > 1e-4 else omega_cmd) / self.p.scrub_omega_scale)
        )
        Mscrub = -scrub_sign * Mscrub_mag

        FL_net = FL_drive + Frr_left
        FR_net = FR_drive + Frr_right

        # Brez umetnega izravnavanja levo/desno pri voznji naravnost.
        # Frr ostane Frr(lambda) * Fz za vsako stran posebej, zato se asimetricna
        # obremenitev na neravnem terenu lahko pozna kot realen yaw moment.
        Frr_left_apply = Frr_left
        Frr_right_apply = Frr_right

        M_drive_net = 0.5 * b * (FR_net - FL_net)

        v_track_left = v_body - 0.5 * b * yaw_rate
        v_track_right = v_body + 0.5 * b * yaw_rate
        P_left = FL_drive * v_track_left
        P_right = FR_drive * v_track_right
        P_scrub_loss = abs(Mscrub * yaw_rate)
        P_rr_loss = abs(Frr_left * v_track_left) + abs(Frr_right * v_track_right)
        P_signed = P_left + P_right - P_scrub_loss - P_rr_loss
        P_positive = max(P_left, 0.0) + max(P_right, 0.0) + P_scrub_loss + P_rr_loss

        traction_scale = self._clip(
            0.5 * (abs(FL_drive) / FL_cap + abs(FR_drive) / FR_cap),
            0.0,
            1.2
        )

        self._last = {
            "v_cmd": float(vcmd),
            "omega_cmd": float(omega_cmd),
            "v_cmd_raw": float(vcmd_raw),
            "omega_cmd_raw": float(omega_raw),
            "v_ref": float(v_ref),
            "omega_ref": float(omega_ref),
            "v_body": float(v_body),
            "yaw_rate": float(yaw_rate),
            "sigma_v": float(sigma_v),
            "sigma_omega": float(sigma_omega),
            "mu": float(mu),
            "lambda": float(lam),
            "Fz_left_N": float(Fz_left),
            "Fz_right_N": float(Fz_right),
            "Fz_total_N": float(Fz_total),
            "FL_drive_N": float(FL_drive),
            "FR_drive_N": float(FR_drive),
            "Frr_left_N": float(Frr_left),
            "Frr_right_N": float(Frr_right),
            "Frr_left_apply_N": float(Frr_left_apply),
            "Frr_right_apply_N": float(Frr_right_apply),
            "FL_net_N": float(FL_net),
            "FR_net_N": float(FR_net),
            "F_total_target_N": float(F_total_target),
            "M_target_Nm": float(M_target),
            "M_scrub_feedforward_Nm": float(M_scrub_feedforward),
            "M_drive_net_Nm": float(M_drive_net),
            "Mscrub_Nm": float(Mscrub),
            "M_total_Nm": float(M_drive_net + Mscrub),
            "crr": float(crr),
            "traction_scale": float(traction_scale),
            "P_left_W": float(P_left),
            "P_right_W": float(P_right),
            "P_scrub_loss_W": float(P_scrub_loss),
            "P_rr_loss_W": float(P_rr_loss),
            "P_signed_total_W": float(P_signed),
            "P_positive_total_W": float(P_positive),
        }
        return dict(self._last)

    def get_state(self):
        return dict(self._last)



class TankSimulator:
    """
    MuJoCo simulator tanka z naslednjimi lastnostmi:
    - Fizika tece v loceni niti
    - GUI (tkinter) tece v glavni niti
    - 3D trajektorija se izrise neposredno v MuJoCo prikazovalniku
    - Dodani so enostavni parametri podlage glede na polozaj (x, y)
    """

    # Nacini voznje, da ne uporabljamo povsod navadnih stringov
    MODE_NONE = None
    MODE_STRAIGHT = "straight"
    MODE_PIVOT = "pivot"
    MODE_CIRCLE = "circle"
    MODE_FIGURE8 = "figure8"

    # Barva trajektorije v obliki RGBA = rdeca, zelena, modra, prosojnost
    TRAJ_RGBA = np.array([0.0, 0.5, 1.0, 0.8], dtype=np.float32)

    def __init__(self, xml_path: str):
        self._experiment_name = Path(xml_path).stem

        # Nalozi MuJoCo model iz XML datoteke
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))

        # Ustvari podatkovno strukturo simulacije za ta model
        self.data = mujoco.MjData(self.model)

        # Stevec za redkejse osvezevanje izrisa trajektorije
        # Ce bi risali vsak korak, bi bilo pocasneje
        self._traj_draw_counter = 0

        # Navaden lock za skupne Python podatke
        self._lock = threading.Lock()

        # Poseben lock za varen dostop do MuJoCo model/data/viewer iz vec niti
        self._mj_lock = threading.Lock()

        # Trenutni nacin voznje
        self._mode = self.MODE_NONE

        # Cas simulacije ob zacetku voznje
        self._drive_start_t = 0.0

        # Maksimalno trajanje posamezne voznje v sekundah simulacije
        self._drive_duration = 45.0

        # Zacetni polozaj vozila ob zacetku voznje
        self._start_pos = None

        # Zacetni yaw kot vozila
        self._start_yaw = 0.0

        # Prejsnji yaw za sprotno racunanje spremembe orientacije
        self._prev_yaw = 0.0

        # Skupna zasukana kotna pot v radianih
        self._total_rot_rad = 0.0

        # Seznama x in y tock trajektorije
        self._traj_x = []
        self._traj_y = []

        # Ali physics nit trenutno tece
        self._physics_running = False

        # Objekt niti fizike
        self._physics_thread = None

        # Viewer objekt za 3D prikaz
        self._viewer = None

        # Tkinter root okno za gumbe
        self._root = None

        # Zadnji material, da ne printamo iste informacije v vsakem koraku
        self._last_material_id = None

        self._pending_stats = None
        self._show_stats_requested = False
      
        base = Path(__file__).resolve().parent
        self._material_map = self._load_material_map(base / "terrains" / "material_map.png")

        self._terrain_world_xmin = -30.0
        self._terrain_world_xmax = 30.0
        self._terrain_world_ymin = -30.0
        self._terrain_world_ymax = 30.0

        # --- PRESTAVLJANJE TANKA Z MISKO ---
        # Python passive MuJoCo viewer nima preprostega javnega callbacka za
        # klik/pick v 3D oknu, zato dodamo prakticen top-down zemljevid terena.
        # Na zemljevidu lahko tank primemo z misko in ga spustimo na novo mesto.
        self._placement_window = None
        self._placement_canvas = None
        self._placement_status_var = None
        self._placement_map_photo = None
        self._placement_canvas_size = 520
        self._placement_dragging = False
        self._placement_preview_xy = None
        self._placement_last_redraw = 0.0

        # Naslov free jointa sasije. Tako ni trdo zakodirano, da je vedno qpos[0].
        try:
            root_jid = self.model.joint("root").id
            self._root_qposadr = int(self.model.jnt_qposadr[root_jid])
            self._root_dofadr = int(self.model.jnt_dofadr[root_jid])
        except Exception:
            self._root_qposadr = 0
            self._root_dofadr = 0

        # Koliko nad terenom naj se postavi referencna tocka sasije.
        # 0.9 m je enako kot zacetni XML polozaj, zato se vozilo po spustu
        # normalno usede na teren.
        self._placement_z_clearance = 0.90

        # --- ANALIZA KONTAKTOV ---
        # Zapisujemo vsoto normalnih kontaktnih sil po levi/desni strani.
        # Ker je timestep 0.002 s, bi vsak korak pomenil 500 vzorcev/s;
        # zato vzamemo vsak 5. korak = približno 100 vzorcev/s.
        self._contact_log = []
        self._contact_point_log = []   # podrobni zapis: ena vrstica za vsak posamezen kontakt
        self._contact_sample_counter = 0
        self._contact_sample_every = 5
        self._wheel_names = [f"L{i}" for i in range(1, 7)] + [f"D{i}" for i in range(1, 7)]

        # --- G4: TERRAMECHANICS / SLIP CORE ---
        # Stara verzija je ukaz samo zmanjsala in ga poslala v actuator ctrl.
        # G4 verzija racuna dejanske sile FL/FR, rolling resistance in scrub moment,
        # potem pa jih v MuJoCo vnese kot zunanje sile/momente na sasijo.
        self._use_terramechanics_core = True
        # Malo visja hitrost je potrebna, da krog in osmica v casu simulacije
        # zakljucita celoten manever tudi po upostevanju zdrsa.
        self._max_track_speed_mps = 3.5

        # Brez yaw-hold stabilizacije: pri NARAVNOST ostane omega_cmd = 0.0.
        # Ce vozilo zaradi realnih kontaktov, neravnega terena ali asimetricnih
        # normalnih sil pocasi zavije, tega namenoma ne popravljamo.

        self._vehicle_mass_kg = max(1.0, float(np.sum(self.model.body_mass)))
        self._track_width_m = self._estimate_track_width()
        self._yaw_inertia_kgm2 = max(
            0.20 * self._vehicle_mass_kg * self._track_width_m * self._track_width_m,
            1.0
        )
        self._terra_core = TerramechanicsCore(TerramechanicsParams(
            mass_kg=self._vehicle_mass_kg,
            track_width_m=self._track_width_m,
            yaw_inertia_kgm2=self._yaw_inertia_kgm2,
            max_side_force_N=max(220.0, 0.95 * self._vehicle_mass_kg * 9.81)
        ))
        self._last_terramech = None

        # G4 pogon: mentorjeva pripomba je bila, da v XML dodamo SITE tocke
        # in silo podamo normalno preko data.ctrl. Zato koda najprej uporabi
        # site aktuatorje drive_L_force, drive_R_force in scrub_yaw_torque.
        # Ce stari XML teh aktuatorjev se nima, ostane varen fallback z mj_applyFT.
        self._body_sasija_id = self.model.body("sasija").id
        self._sasija = self.data.body("sasija")
        self._ctrl = self.data.ctrl                 # pogled na MjData ctrl; ne delamo ga znova v zanki
        self._qpos = self.data.qpos                 # pogled na MjData qpos
        self._qvel = self.data.qvel                 # pogled na MjData qvel
        self._qfrc_applied = self.data.qfrc_applied # pogled na MjData qfrc_applied
        self._site_xpos = self.data.site_xpos       # pogled na pozicije vseh site-ov
        self._drive_site_L_id = self._try_get_site_id("drive_L")
        self._drive_site_R_id = self._try_get_site_id("drive_R")
        self._drive_site_C_id = self._try_get_site_id("drive_C")
        self._drive_actuator_L_id = self._try_get_actuator_id("drive_L_force")
        self._drive_actuator_R_id = self._try_get_actuator_id("drive_R_force")
        self._scrub_actuator_id = self._try_get_actuator_id("scrub_yaw_torque")
        self._site_actuator_drive = (
            self._drive_actuator_L_id is not None
            and self._drive_actuator_R_id is not None
            and self._scrub_actuator_id is not None
        )
        if self._site_actuator_drive:
            print("G4 pogon: uporabljam SITE aktuatorje drive_L_force/drive_R_force/scrub_yaw_torque.")
        else:
            print("OPOZORILO: SITE aktuatorji niso najdeni; uporabljam fallback mj_applyFT na sasijo.")

        # --- ANALIZA EFEKTIVNEGA ZDRSA ---
        # Zdrs tukaj ni dodan kot poseben MuJoCo element v XML,
        # ampak ga uporabimo kot kinematični popravek ukazov pred mj_step.
        # sigma_v zmanjša translacijski del diferencialnega pogona,
        # sigma_omega pa rotacijski del diferencialnega pogona.
        self._slip_log = []
        self._slip_sample_counter = 0

        # --- OCENA ENERGIJE POGONA ---
        # Mehansko moc pogona ocenimo iz MuJoCo actuator_force in actuator_velocity:
        #   P_i = F_actuator_i * v_actuator_i
        # Pri rotacijskem motorju to pomeni priblizno:
        #   P_i = tau_i * omega_i
        # Energijo dobimo z numericnim integralom:
        #   E = integral(P dt) ~= vsota(P_k * dt)
        # Za porabo je najbolj uporabna pozitivna moc po aktuatorjih, ker negativna
        # moc pomeni zaviranje/vracanje energije in ne sme izniciti porabe drugih motorjev.
        self._energy_log = []
        self._energy_sample_counter = 0
        self._energy_signed_J = 0.0
        self._energy_positive_J = 0.0
        self._energy_abs_J = 0.0
        self._energy_per_actuator_positive_J = np.zeros(self.model.nu, dtype=float)
        self._energy_per_actuator_abs_J = np.zeros(self.model.nu, dtype=float)
        self._actuator_names = []
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                i
            )
            self._actuator_names.append(name or f"actuator_{i}")

        # Tvoj neravni teren se imenuje "terrain", raven teren pa "povrsina".
        # Tako ista koda dela za oba XML-ja.
        self._terrain_geom_ids = set()
        for geom_name in ("terrain", "povrsina"):
            try:
                self._terrain_geom_ids.add(self.model.geom(geom_name).id)
            except Exception:
                pass

        # Pri G4 force-injection naj MuJoCo ne generira se enega locenega
        # longitudinalnega pogona prek kontaktnega trenja. Normalne sile ostanejo,
        # tangencialno trenje pa znižamo, da se izognemo dvojnemu trenju in nihanju.
        self._configure_force_injection_contact_friction()

    def _load_material_map(self, path):
        img = Image.open(str(path)).convert("RGB")
        return np.array(img, dtype=np.uint8)

    def _configure_force_injection_contact_friction(self):
        """
        Zmanjsa MuJoCo tangencialno trenje na kolesih/terenu pri G4 force-injection.

        Razlog: FL/FR, rolling resistance in scrub torque so zdaj izracunani v
        TerramechanicsCore. Ce MuJoCo hkrati uporablja veliko kontaktno trenje, dobimo
        dvojno trenje: MuJoCo + G4. To je eden glavnih razlogov za trzanje ali
        pocasno-hitro-pocasno obnasanje pri pivotu.
        """
        if not self._use_terramechanics_core:
            return

        target_friction = np.array([0.22, 0.006, 0.0002], dtype=float)
        geom_ids = set(int(g) for g in self._terrain_geom_ids)

        # Vsa kolesa/podporna telesa L1..L6 in D1..D6.
        for body_name in self._wheel_names:
            try:
                bid = int(self.model.body(body_name).id)
                adr = int(self.model.body_geomadr[bid])
                num = int(self.model.body_geomnum[bid])
                for gid in range(adr, adr + num):
                    geom_ids.add(int(gid))
            except Exception:
                pass

        for gid in geom_ids:
            try:
                self.model.geom_friction[gid, :3] = target_friction
            except Exception:
                pass


    def _try_get_site_id(self, site_name: str):
        """Vrne id site-a ali None, ce site v XML modelu ne obstaja."""
        try:
            return int(self.model.site(site_name).id)
        except Exception:
            return None

    def _try_get_actuator_id(self, actuator_name: str):
        """Vrne id aktuatorja ali None, ce aktuator v XML modelu ne obstaja."""
        try:
            return int(self.model.actuator(actuator_name).id)
        except Exception:
            return None

    def _estimate_track_width(self) -> float:
        """
        Oceni efektivno sirino med levo in desno sledjo iz imen teles L1..L6 in D1..D6.
        Ce tega ne more, uporabi varno privzeto vrednost.
        """
        left_y = []
        right_y = []

        for name in [f"L{i}" for i in range(1, 7)]:
            try:
                bid = int(self.model.body(name).id)
                left_y.append(float(self.model.body_pos[bid][1]))
            except Exception:
                pass

        for name in [f"D{i}" for i in range(1, 7)]:
            try:
                bid = int(self.model.body(name).id)
                right_y.append(float(self.model.body_pos[bid][1]))
            except Exception:
                pass

        if left_y and right_y:
            width = abs(float(np.mean(left_y)) - float(np.mean(right_y)))
            if width > 0.05:
                return width

        # Fallback: poskusimo iz geometrije sasije.
        try:
            gid = int(self.model.geom("sasija").id)
            size = np.asarray(self.model.geom_size[gid], dtype=float)
            if size.size >= 2 and float(size[1]) > 0.05:
                return float(2.0 * size[1])
        except Exception:
            pass

        return 1.00

    def _body_frame_axes_locked(self):
        """
        Vrne osi sasije v svetovnem koordinatnem sistemu:
        forward_world, left_world, up_world.
        """
        # self._sasija.xmat je ze NumPy pogled v MjData; ne pretvarjamo ga v nov array.
        R = self._sasija.xmat.reshape(3, 3)
        forward_world = R[:, 0].copy()
        left_world = R[:, 1].copy()
        up_world = R[:, 2].copy()
        return forward_world, left_world, up_world

    def _body_planar_state_locked(self):
        """
        Prebere trenutno vzdolzno hitrost in yaw rate iz MuJoCo.

        Uporabi mj_objectVelocity, ker se s tem izognemo ugibanju o indeksih freejoint qvel.
        """
        body_id = int(self._body_sasija_id)
        vel6 = np.zeros(6, dtype=np.float64)

        try:
            mujoco.mj_objectVelocity(
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
                vel6,
                0  # 0 = svetovni koordinatni sistem
            )
            ang_world = vel6[:3]
            lin_world = vel6[3:]
        except Exception:
            # Fallback za primer starejsega MuJoCo API-ja.
            qv = self._qvel
            dadr = int(self._root_dofadr)
            lin_world = qv[dadr:dadr + 3] if qv.size >= dadr + 3 else np.zeros(3)
            ang_world = qv[dadr + 3:dadr + 6] if qv.size >= dadr + 6 else np.zeros(3)

        forward_world, _left_world, up_world = self._body_frame_axes_locked()
        v_forward = float(np.dot(lin_world, forward_world))
        yaw_rate = float(np.dot(ang_world, up_world))

        return {
            "v_forward": v_forward,
            "yaw_rate": yaw_rate,
        }

    def _compute_contact_loads_locked(self):
        """
        Vrne sprotno vsoto normalnih sil po levi/desni strani.

        To je isti vir podatkov kot pri analizi kontaktov, samo brez redkega vzorcenja
        in brez zapisovanja v CSV. G4 to uporablja kot vhod loads.
        """
        f_left = 0.0
        f_right = 0.0
        n_left = 0
        n_right = 0

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1 = int(con.geom1)
            g2 = int(con.geom2)

            if g1 in self._terrain_geom_ids and g2 not in self._terrain_geom_ids:
                wheel_geom = g2
            elif g2 in self._terrain_geom_ids and g1 not in self._terrain_geom_ids:
                wheel_geom = g1
            else:
                continue

            side, _body_name = self._side_from_geom_id(wheel_geom)
            if side is None:
                continue

            force6 = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, force6)
            fn = max(0.0, float(force6[0]))

            if side == "L":
                f_left += fn
                n_left += 1
            elif side == "D":
                f_right += fn
                n_right += 1

        if f_left + f_right < 1.0:
            # Prvi korak po resetu lahko se nima kontaktov. Uporabimo staticno oceno.
            f_total = max(1.0, self._vehicle_mass_kg * 9.81)
            f_left = 0.5 * f_total
            f_right = 0.5 * f_total

        return {
            "Fz_left": float(f_left),
            "Fz_right": float(f_right),
            "n_left": int(n_left),
            "n_right": int(n_right),
        }

    def _left_right_command_to_cmd_vel(self, left_cmd: float, right_cmd: float):
        """
        Pretvori stare GUI ukaze leve/desne strani v vcmd in omega_cmd.

        To ohrani gumbe NARAVNOST/PIVOT/KROG/OSMICA, notranji G4 pa dobi vhod
        v obliki, ki jo zahtevajo navodila: zelena linearna in kotna hitrost.
        """
        v_left = float(left_cmd) * self._max_track_speed_mps
        v_right = float(right_cmd) * self._max_track_speed_mps
        b = max(0.05, float(self._track_width_m))

        v_cmd = 0.5 * (v_left + v_right)
        omega_cmd = (v_right - v_left) / b

        return {
            "v_cmd": float(v_cmd),
            "omega_cmd": float(omega_cmd),
            "v_left_cmd_mps": float(v_left),
            "v_right_cmd_mps": float(v_right),
        }

    def _apply_terramechanics_forces_locked(self, drive_info: dict):
        """
        Vnese G4 sile v MuJoCo.

        Primarna pot po mentorjevem predlogu:
          - XML ima site-e drive_L, drive_R, drive_C,
          - XML ima motor aktuatorje na teh site-ih,
          - sile/moment se nastavijo z data.ctrl.

        Fallback ostane samo zato, da se program ne zlomi, ce kdo pozenes star XML.
        """
        if drive_info is None:
            self._ctrl[:] = 0.0
            self._qfrc_applied[:] = 0.0
            return

        # Ne puscamo starih zunanjih sil iz prejsnjega koraka.
        self._qfrc_applied[:] = 0.0

        FL = float(drive_info.get("FL_net_N", 0.0))
        FR = float(drive_info.get("FR_net_N", 0.0))
        Mscrub = float(drive_info.get("Mscrub_Nm", 0.0))

        if self._site_actuator_drive:
            # Mentorjeva zahteva: site + data.ctrl.
            # XML gear lokalni +X site-a sam pretvori v svetovni sistem glede na orientacijo sasije.
            self._ctrl[self._drive_actuator_L_id] = FL
            self._ctrl[self._drive_actuator_R_id] = FR
            self._ctrl[self._scrub_actuator_id] = Mscrub
            return

        # Fallback: ekvivalentna aplikacija sile/momenta na sasijo z mj_applyFT.
        # Tega se pri popravljenem XML-ju ne uporablja.
        forward_world, _left_world, up_world = self._body_frame_axes_locked()
        body_id = int(self._body_sasija_id)

        if self._drive_site_L_id is not None and self._drive_site_R_id is not None:
            pL = self._site_xpos[self._drive_site_L_id].copy()
            pR = self._site_xpos[self._drive_site_R_id].copy()

            mujoco.mj_applyFT(
                self.model,
                self.data,
                forward_world * FL,
                np.zeros(3, dtype=np.float64),
                pL,
                body_id,
                self._qfrc_applied
            )
            mujoco.mj_applyFT(
                self.model,
                self.data,
                forward_world * FR,
                np.zeros(3, dtype=np.float64),
                pR,
                body_id,
                self._qfrc_applied
            )

            pC = self._sasija.xpos.copy()
            mujoco.mj_applyFT(
                self.model,
                self.data,
                np.zeros(3, dtype=np.float64),
                up_world * Mscrub,
                pC,
                body_id,
                self._qfrc_applied
            )
        else:
            total_force = forward_world * (FL + FR)
            total_torque = up_world * float(drive_info.get("M_total_Nm", 0.0))
            pC = self._sasija.xpos.copy()

            mujoco.mj_applyFT(
                self.model,
                self.data,
                total_force,
                total_torque,
                pC,
                body_id,
                self._qfrc_applied
            )


    @staticmethod
    def _quat_to_yaw(q: np.ndarray) -> float:
        """Pretvori MuJoCo kvaternion [w, x, y, z] v yaw."""
        w, x, y, z = q                    # Razpakiramo komponente kvaterniona
        return float(np.arctan2(         # Izracun yaw kota okoli navpicne osi
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z)
        ))
    
    @staticmethod
    def _quat_to_rpy(q: np.ndarray):
        """Pretvori MuJoCo kvaternion [w, x, y, z] v roll, pitch, yaw [rad]."""
        w, x, y, z = q

        roll = np.arctan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (x * x + y * y)
        )

        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)

        yaw = np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z)
        )

        return float(roll), float(pitch), float(yaw)
    @staticmethod
    def _wrap_angle(a: float) -> float:
        """Normalizira kot na interval [-pi, pi]."""
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)

    def terrain_params(self, x: float, y: float):
        """
        Vrne (mu, lambda, material_id, material_name)
        glede na material_map sliko.
        """

        img = self._material_map
        h, w, _ = img.shape

        # svetovne koordinate -> [0, 1]
        u = (x - self._terrain_world_xmin) / (self._terrain_world_xmax - self._terrain_world_xmin)
        v = (y - self._terrain_world_ymin) / (self._terrain_world_ymax - self._terrain_world_ymin)

        # omejitev na robove
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))

        # [0,1] -> piksel
        px = int(u * (w - 1))
        py = int((1.0 - v) * (h - 1))

        r, g, b = img[py, px]

        #  blato/zemlja = rdeca
        if r > 200 and g < 80 and b < 80:
             return 0.8, 0.5, 1, "blato"
            

        #asfalt = zelena
        elif g > 200 and r < 80 and b < 80:
            return 1.2, 0.2, 0, "asfalt"
           
        # pesek = modra
        elif b > 200 and r < 80 and g < 80:
            return 0.5, 0.9, 2, "pesek"

        # fallback
        return 0.8, 0.5, 1, "blato"

    def _slip_params(self, mu: float, lam: float, mat_name: str):
        """
        Vrne efektivna koeficienta zdrsa (sigma_v, sigma_omega).

        Ideja iz porocila:
            v     = v_ideal     * (1 - sigma_v)
            omega = omega_ideal * (1 - sigma_omega)

        V tej poenostavljeni verziji koeficienta izpeljemo iz materiala podlage:
        - vecji lambda pomeni vec zdrsa,
        - manjsi mu malo poveca zdrs,
        - sigma_omega je navadno vecji, ker vozilo pri zavijanju/pivotiranju
          lazje zdrsne bocno kot pri voznji naravnost.
        """
        # osnovni vpliv materiala
        mu01 = max(0.0, min(1.0, float(mu)))
        lam01 = max(0.0, min(1.0, float(lam)))

        sigma_v = 0.80 * lam01 + 0.10 * (1.0 - mu01)
        sigma_omega = 1.00 * lam01 + 0.15 * (1.0 - mu01)

        # razumne meje: 0 pomeni brez zdrsa, 0.95 skoraj popoln zdrs
        sigma_v = max(0.0, min(0.95, sigma_v))
        sigma_omega = max(0.0, min(0.95, sigma_omega))

        return float(sigma_v), float(sigma_omega)

    def _apply_effective_slip_to_commands(self, left_cmd: float, right_cmd: float,
                                          mu: float, lam: float, mat_name: str):
        """
        Diferencialni pogon razdelimo na translacijski in rotacijski del:
            v_cmd     = (left + right) / 2
            omega_cmd = (right - left) / 2

        Nato uporabimo zdrs:
            v_eff     = v_cmd     * (1 - sigma_v)
            omega_eff = omega_cmd * (1 - sigma_omega)

        Na koncu pretvorimo nazaj v levi/desni ukaz motorjev.
        Mu ne predstavlja zdrsa, ampak grobo spremembo ucinkovitosti/oprimka podlage.
        """
        sigma_v, sigma_omega = self._slip_params(mu, lam, mat_name)

        v_cmd = 0.5 * (float(left_cmd) + float(right_cmd))
        omega_cmd = 0.5 * (float(right_cmd) - float(left_cmd))

        v_eff = v_cmd * (1.0 - sigma_v)
        omega_eff = omega_cmd * (1.0 - sigma_omega)

        # mu uporabimo kot dodatni faktor ucinkovitosti pogona.
        drive_gain = max(0.05, min(1.2, float(mu)))

        left_eff = (v_eff - omega_eff) * drive_gain
        right_eff = (v_eff + omega_eff) * drive_gain

        # Primerjalni faktor za stare grafe in izpise.
        traction_scale = drive_gain * (1.0 - 0.5 * (sigma_v + sigma_omega))
        traction_scale = max(0.05, min(1.2, traction_scale))

        return {
            "left_eff": float(left_eff),
            "right_eff": float(right_eff),
            "v_cmd_ideal": float(v_cmd),
            "omega_cmd_ideal": float(omega_cmd),
            "v_cmd_eff": float(v_eff),
            "omega_cmd_eff": float(omega_eff),
            "sigma_v_eff": float(sigma_v),
            "sigma_omega_eff": float(sigma_omega),
            "drive_gain_mu": float(drive_gain),
            "traction_scale": float(traction_scale),
        }

    def _sample_slip_locked(self, mode: str, mat_name: str, mu: float, lam: float,
                            drive_info: dict):
        """Shrani efektivni zdrs skozi cas."""
        if mode is None or drive_info is None:
            return

        self._slip_sample_counter += 1
        if self._slip_sample_counter % self._contact_sample_every != 0:
            return

        pos = self.data.body("sasija").xpos
        row = {
            "t": float(self.data.time - self._drive_start_t),
            "sim_time": float(self.data.time),
            "mode": str(mode),
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
            "material": str(mat_name),
            "mu": float(mu),
            "lambda": float(lam),
            "sigma_v_eff": float(drive_info["sigma_v_eff"]),
            "sigma_omega_eff": float(drive_info["sigma_omega_eff"]),
            "v_cmd_ideal": float(drive_info["v_cmd_ideal"]),
            "omega_cmd_ideal": float(drive_info["omega_cmd_ideal"]),
            "v_cmd_eff": float(drive_info["v_cmd_eff"]),
            "omega_cmd_eff": float(drive_info["omega_cmd_eff"]),
            "left_cmd_eff": float(drive_info["left_eff"]),
            "right_cmd_eff": float(drive_info["right_eff"]),
            "drive_gain_mu": float(drive_info["drive_gain_mu"]),
            "traction_scale": float(drive_info["traction_scale"]),
            "Fz_left_N": float(drive_info.get("Fz_left_N", 0.0)),
            "Fz_right_N": float(drive_info.get("Fz_right_N", 0.0)),
            "FL_drive_N": float(drive_info.get("FL_drive_N", 0.0)),
            "FR_drive_N": float(drive_info.get("FR_drive_N", 0.0)),
            "Frr_left_N": float(drive_info.get("Frr_left_N", 0.0)),
            "Frr_right_N": float(drive_info.get("Frr_right_N", 0.0)),
            "FL_net_N": float(drive_info.get("FL_net_N", 0.0)),
            "FR_net_N": float(drive_info.get("FR_net_N", 0.0)),
            "M_drive_net_Nm": float(drive_info.get("M_drive_net_Nm", 0.0)),
            "Mscrub_Nm": float(drive_info.get("Mscrub_Nm", 0.0)),
            "M_total_Nm": float(drive_info.get("M_total_Nm", 0.0)),
            "crr": float(drive_info.get("crr", 0.0)),
        }
        with self._lock:
            self._slip_log.append(row)

    def _actuator_velocity_vector_locked(self):
        """
        Vrne vektor hitrosti aktuatorjev.

        V novejsih verzijah MuJoCo je to neposredno v data.actuator_velocity.
        Ce tega polja ni, uporabimo fallback za motorje na hinge jointih:
        actuator_velocity ~= gear * qvel[joint_dof].
        """
        if hasattr(self.data, "actuator_velocity"):
            return np.asarray(self.data.actuator_velocity, dtype=float).copy()

        # Fallback 1: ce je na voljo matrika momentnih rocic aktuatorjev.
        if hasattr(self.data, "actuator_moment"):
            moment = np.asarray(self.data.actuator_moment, dtype=float).reshape(self.model.nu, self.model.nv)
            return moment @ np.asarray(self.data.qvel, dtype=float)

        # Fallback 2: za navadne motorje na jointih.
        vel = np.zeros(self.model.nu, dtype=float)
        for i in range(self.model.nu):
            try:
                joint_id = int(self.model.actuator_trnid[i, 0])
                if joint_id < 0:
                    continue
                dof_adr = int(self.model.jnt_dofadr[joint_id])
                gear = float(self.model.actuator_gear[i, 0])
                vel[i] = gear * float(self.data.qvel[dof_adr])
            except Exception:
                vel[i] = 0.0
        return vel

    def _sample_energy_locked(self, mode: str, mat_name: str, drive_info: dict, dt: float):
        """
        Oceni moc in energijo pogona.

        V G4 verziji je primarni vir moci terramechanics core:
            P ~= FL * vL + FR * vR + izgube zaradi rolling resistance/scrub.

        Ce G4 podatkov ni, ostane fallback na MuJoCo actuator_force * actuator_velocity.
        """
        if mode is None:
            return

        dt = float(dt)

        # --- G4 vir energije: sile FL/FR + upori ---
        if drive_info is not None and "P_positive_total_W" in drive_info:
            p_signed_total = float(drive_info.get("P_signed_total_W", 0.0))
            p_positive_total = float(drive_info.get("P_positive_total_W", 0.0))
            p_abs_total = abs(p_signed_total) + float(drive_info.get("P_scrub_loss_W", 0.0)) + float(drive_info.get("P_rr_loss_W", 0.0))

            p_left_signed = float(drive_info.get("P_left_W", 0.0))
            p_right_signed = float(drive_info.get("P_right_W", 0.0))
            p_left_positive = max(p_left_signed, 0.0)
            p_right_positive = max(p_right_signed, 0.0)

            # Pri SITE aktuatorjih energijo pripisemo tudi posameznim novim
            # aktuatorjem, da graf "energija_po_motorjih" ni vec prazen.
            if self._drive_actuator_L_id is not None:
                self._energy_per_actuator_positive_J[self._drive_actuator_L_id] += p_left_positive * dt
                self._energy_per_actuator_abs_J[self._drive_actuator_L_id] += abs(p_left_signed) * dt
            if self._drive_actuator_R_id is not None:
                self._energy_per_actuator_positive_J[self._drive_actuator_R_id] += p_right_positive * dt
                self._energy_per_actuator_abs_J[self._drive_actuator_R_id] += abs(p_right_signed) * dt
            if self._scrub_actuator_id is not None:
                scrub_loss = float(drive_info.get("P_scrub_loss_W", 0.0))
                self._energy_per_actuator_positive_J[self._scrub_actuator_id] += scrub_loss * dt
                self._energy_per_actuator_abs_J[self._scrub_actuator_id] += scrub_loss * dt

            self._energy_signed_J += p_signed_total * dt
            self._energy_positive_J += p_positive_total * dt
            self._energy_abs_J += p_abs_total * dt

            self._energy_sample_counter += 1
            if self._energy_sample_counter % self._contact_sample_every != 0:
                return

            pos = self.data.body("sasija").xpos
            row = {
                "t": float(self.data.time - self._drive_start_t),
                "sim_time": float(self.data.time),
                "mode": str(mode),
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
                "material": str(mat_name),
                "P_signed_total_W": p_signed_total,
                "P_positive_total_W": p_positive_total,
                "P_abs_total_W": p_abs_total,
                "P_signed_left_W": p_left_signed,
                "P_signed_right_W": p_right_signed,
                "P_positive_left_W": p_left_positive,
                "P_positive_right_W": p_right_positive,
                "P_scrub_loss_W": float(drive_info.get("P_scrub_loss_W", 0.0)),
                "P_rr_loss_W": float(drive_info.get("P_rr_loss_W", 0.0)),
                "E_signed_total_J": float(self._energy_signed_J),
                "E_positive_total_J": float(self._energy_positive_J),
                "E_abs_total_J": float(self._energy_abs_J),
                "E_positive_total_Wh": float(self._energy_positive_J / 3600.0),
                "E_abs_total_Wh": float(self._energy_abs_J / 3600.0),
                "sigma_v_eff": float(drive_info.get("sigma_v_eff", 0.0)),
                "sigma_omega_eff": float(drive_info.get("sigma_omega_eff", 0.0)),
                "left_cmd_eff": float(drive_info.get("left_eff", 0.0)),
                "right_cmd_eff": float(drive_info.get("right_eff", 0.0)),
                "Fz_left_N": float(drive_info.get("Fz_left_N", 0.0)),
                "Fz_right_N": float(drive_info.get("Fz_right_N", 0.0)),
                "FL_drive_N": float(drive_info.get("FL_drive_N", 0.0)),
                "FR_drive_N": float(drive_info.get("FR_drive_N", 0.0)),
                "Frr_left_N": float(drive_info.get("Frr_left_N", 0.0)),
                "Frr_right_N": float(drive_info.get("Frr_right_N", 0.0)),
                "FL_net_N": float(drive_info.get("FL_net_N", 0.0)),
                "FR_net_N": float(drive_info.get("FR_net_N", 0.0)),
                "Mscrub_Nm": float(drive_info.get("Mscrub_Nm", 0.0)),
                "M_total_Nm": float(drive_info.get("M_total_Nm", 0.0)),
                "crr": float(drive_info.get("crr", 0.0)),
            }

            with self._lock:
                self._energy_log.append(row)
            return

        # --- Fallback: stara ocena prek aktuatorjev ---
        if self.model.nu == 0:
            return

        actuator_force = np.asarray(self.data.actuator_force, dtype=float).copy()
        actuator_velocity = self._actuator_velocity_vector_locked()

        if actuator_force.shape[0] != self.model.nu or actuator_velocity.shape[0] != self.model.nu:
            return

        power = actuator_force * actuator_velocity
        power_positive = np.maximum(power, 0.0)
        power_abs = np.abs(power)

        p_signed_total = float(np.sum(power))
        p_positive_total = float(np.sum(power_positive))
        p_abs_total = float(np.sum(power_abs))

        left_idx = [i for i, name in enumerate(self._actuator_names) if name.startswith("L")]
        right_idx = [i for i, name in enumerate(self._actuator_names) if name.startswith("D")]

        p_left_signed = float(np.sum(power[left_idx])) if left_idx else 0.0
        p_right_signed = float(np.sum(power[right_idx])) if right_idx else 0.0
        p_left_positive = float(np.sum(power_positive[left_idx])) if left_idx else 0.0
        p_right_positive = float(np.sum(power_positive[right_idx])) if right_idx else 0.0

        self._energy_signed_J += p_signed_total * dt
        self._energy_positive_J += p_positive_total * dt
        self._energy_abs_J += p_abs_total * dt
        self._energy_per_actuator_positive_J += power_positive * dt
        self._energy_per_actuator_abs_J += power_abs * dt

        self._energy_sample_counter += 1
        if self._energy_sample_counter % self._contact_sample_every != 0:
            return

        pos = self.data.body("sasija").xpos
        row = {
            "t": float(self.data.time - self._drive_start_t),
            "sim_time": float(self.data.time),
            "mode": str(mode),
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
            "material": str(mat_name),
            "P_signed_total_W": p_signed_total,
            "P_positive_total_W": p_positive_total,
            "P_abs_total_W": p_abs_total,
            "P_signed_left_W": p_left_signed,
            "P_signed_right_W": p_right_signed,
            "P_positive_left_W": p_left_positive,
            "P_positive_right_W": p_right_positive,
            "E_signed_total_J": float(self._energy_signed_J),
            "E_positive_total_J": float(self._energy_positive_J),
            "E_abs_total_J": float(self._energy_abs_J),
            "E_positive_total_Wh": float(self._energy_positive_J / 3600.0),
            "E_abs_total_Wh": float(self._energy_abs_J / 3600.0),
        }

        if drive_info is not None:
            row.update({
                "sigma_v_eff": float(drive_info["sigma_v_eff"]),
                "sigma_omega_eff": float(drive_info["sigma_omega_eff"]),
                "left_cmd_eff": float(drive_info["left_eff"]),
                "right_cmd_eff": float(drive_info["right_eff"]),
            })

        for i, name in enumerate(self._actuator_names):
            row[f"force_{name}"] = float(actuator_force[i])
            row[f"velocity_{name}"] = float(actuator_velocity[i])
            row[f"power_{name}_W"] = float(power[i])
            row[f"energy_positive_{name}_J"] = float(self._energy_per_actuator_positive_J[i])
            row[f"energy_abs_{name}_J"] = float(self._energy_per_actuator_abs_J[i])

        with self._lock:
            self._energy_log.append(row)


    def _side_from_geom_id(self, geom_id: int):
        """
        Iz geometrije kontakta najde telo, na katerem je geom.
        Kolesa so v XML-ju v telesih L1..L6 in D1..D6.
        Vrne: ("L" ali "D" ali None, ime_telesa)
        """
        body_id = int(self.model.geom_bodyid[int(geom_id)])
        body_name = mujoco.mj_id2name(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id
        ) or ""

        if body_name.startswith("L"):
            return "L", body_name
        if body_name.startswith("D"):
            return "D", body_name
        return None, body_name

    def _sample_contacts_locked(self, mode: str, mat_name: str, traction_scale: float):
        """
        Prebere trenutne MuJoCo kontakte in shrani:
        1) casovni zapis vsote normalnih sil in stevila kontaktov,
        2) podrobni zapis posameznih kontaktnih tock (x, y, z).

        Klicati jo moramo po mujoco.mj_step(...), ko so kontakti izracunani.

        V MuJoCo je force[0] normalna komponenta kontaktne sile v kontaktnem koordinatnem sistemu.
        force[1] in force[2] sta tangencialni komponenti trenja.
        """
        if mode is None:
            return

        self._contact_sample_counter += 1
        if self._contact_sample_counter % self._contact_sample_every != 0:
            return

        f_left = 0.0
        f_right = 0.0
        n_left = 0
        n_right = 0
        per_wheel_force = {name: 0.0 for name in self._wheel_names}
        per_wheel_count = {name: 0 for name in self._wheel_names}
        contact_points = []

        # Preverimo vse kontakte v trenutnem simulacijskem koraku.
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1 = int(con.geom1)
            g2 = int(con.geom2)

            # Upostevamo samo kontakt kolo <-> teren.
            if g1 in self._terrain_geom_ids and g2 not in self._terrain_geom_ids:
                wheel_geom = g2
                terrain_geom = g1
            elif g2 in self._terrain_geom_ids and g1 not in self._terrain_geom_ids:
                wheel_geom = g1
                terrain_geom = g2
            else:
                continue

            side, body_name = self._side_from_geom_id(wheel_geom)
            if side is None:
                continue

            force6 = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, force6)
            fn = max(0.0, float(force6[0]))  # normalna sila kontakta [N]

            # Polozaj kontaktne tocke v svetovnem koordinatnem sistemu.
            cx, cy, cz = [float(v) for v in con.pos]

            if body_name in per_wheel_force:
                per_wheel_force[body_name] += fn
                per_wheel_count[body_name] += 1

            if side == "L":
                f_left += fn
                n_left += 1
            elif side == "D":
                f_right += fn
                n_right += 1

            contact_points.append({
                "t": float(self.data.time - self._drive_start_t),
                "sim_time": float(self.data.time),
                "mode": str(mode),
                "side": side,
                "wheel": body_name,
                "contact_x": cx,
                "contact_y": cy,
                "contact_z": cz,
                "normal_force_N": fn,
                "contact_distance_m": float(con.dist),
                "material": str(mat_name),
                "terrain_geom_id": int(terrain_geom),
                "wheel_geom_id": int(wheel_geom),
            })

        pos = self.data.body("sasija").xpos
        quat = self._qpos[self._root_qposadr + 3:self._root_qposadr + 7]
        roll, pitch, yaw = self._quat_to_rpy(quat)
        row = {
            "t": float(self.data.time - self._drive_start_t),
            "sim_time": float(self.data.time),
            "mode": str(mode),
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
            "roll_deg": float(np.degrees(roll)),
            "pitch_deg": float(np.degrees(pitch)),
            "yaw_deg": float(np.degrees(yaw)),
            "material": str(mat_name),
            "traction_scale": float(traction_scale),
            "F_normal_left_N": f_left,
            "F_normal_right_N": f_right,
            "F_normal_total_N": f_left + f_right,
            "F_normal_left_minus_right_N": f_left - f_right,
            "n_contacts_left": n_left,
            "n_contacts_right": n_right,
            "n_contacts_total": n_left + n_right,
        }
        for wheel_name, value in per_wheel_force.items():
            row[f"F_{wheel_name}_N"] = float(value)
        for wheel_name, value in per_wheel_count.items():
            row[f"n_{wheel_name}"] = int(value)

        with self._lock:
            self._contact_log.append(row)
            self._contact_point_log.extend(contact_points)

    def _save_contact_analysis(self):
        """Shrani CSV in PNG grafe za zadnjo voznjo."""
        with self._lock:
            rows = list(self._contact_log)
            point_rows = list(self._contact_point_log)
            slip_rows = list(self._slip_log)
            energy_rows = list(self._energy_log)
            energy_per_actuator_positive_J = self._energy_per_actuator_positive_J.copy()
            energy_per_actuator_abs_J = self._energy_per_actuator_abs_J.copy()
            energy_positive_total_J = float(self._energy_positive_J)
            energy_abs_total_J = float(self._energy_abs_J)
            energy_signed_total_J = float(self._energy_signed_J)

        if not rows:
            print("Ni kontaktnih podatkov za izris grafov.")
            return None

        base = Path(__file__).resolve().parent
        out_dir = base / "eksperimenti"
        out_dir.mkdir(exist_ok=True)

        mode = rows[0].get("mode", "voznja")
        prefix = f"{self._experiment_name}_{mode}"

        # 1) CSV z zbirnimi podatki po casovnih korakih
        csv_path = out_dir / f"{prefix}_kontaktne_sile.csv"
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # 2) CSV s posameznimi kontaktnimi tockami
        points_csv_path = out_dir / f"{prefix}_kontaktne_tocke.csv"
        if point_rows:
            point_fieldnames = list(point_rows[0].keys())
            with points_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=point_fieldnames)
                writer.writeheader()
                writer.writerows(point_rows)

        # 3) CSV z efektivnim zdrsom
        slip_csv_path = out_dir / f"{prefix}_efektivni_zdrs.csv"
        if slip_rows:
            slip_fieldnames = list(slip_rows[0].keys())
            with slip_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=slip_fieldnames)
                writer.writeheader()
                writer.writerows(slip_rows)

        # 4) CSV z oceno moci in energije pogona
        energy_csv_path = out_dir / f"{prefix}_energija_pogona.csv"
        if energy_rows:
            energy_fieldnames = list(energy_rows[0].keys())
            with energy_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=energy_fieldnames)
                writer.writeheader()
                writer.writerows(energy_rows)

        t = np.array([r["t"] for r in rows], dtype=float)
        fL = np.array([r["F_normal_left_N"] for r in rows], dtype=float)
        fD = np.array([r["F_normal_right_N"] for r in rows], dtype=float)
        fT = np.array([r["F_normal_total_N"] for r in rows], dtype=float)
        fDiff = np.array([r["F_normal_left_minus_right_N"] for r in rows], dtype=float)
        nL = np.array([r["n_contacts_left"] for r in rows], dtype=float)
        nD = np.array([r["n_contacts_right"] for r in rows], dtype=float)
        nT = np.array([r["n_contacts_total"] for r in rows], dtype=float)
        roll_deg = np.array([r.get("roll_deg", 0.0) for r in rows], dtype=float)
        pitch_deg = np.array([r.get("pitch_deg", 0.0) for r in rows], dtype=float)

        # 3) Graf vsote normalnih sil levo/desno/skupaj
        png_forces = out_dir / f"{prefix}_normalne_sile_levo_desno.png"
        plt.figure(figsize=(10, 5))
        plt.plot(t, fL, label="Leva stran")
        plt.plot(t, fD, label="Desna stran")
        plt.plot(t, fT, label="Skupaj")
        plt.xlabel("cas voznje [s]")
        plt.ylabel("normalna sila [N]")
        plt.title("Vsota normalnih kontaktnih sil po levi/desni strani")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_forces, dpi=200)
        plt.close()

        # 4) Graf razlike levo - desno
        png_balance = out_dir / f"{prefix}_razlika_levo_desno.png"
        plt.figure(figsize=(10, 4))
        plt.plot(t, fDiff, label="Leva - desna")
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("cas voznje [s]")
        plt.ylabel("razlika normalne sile [N]")
        plt.title("Porazdelitev obremenitve: leva minus desna stran")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_balance, dpi=200)
        plt.close()

        # 5) Povprecna normalna sila po posameznih kolesih
        wheel_means = []
        for wheel_name in self._wheel_names:
            key = f"F_{wheel_name}_N"
            vals = np.array([r[key] for r in rows], dtype=float)
            wheel_means.append(float(vals.mean()))

        png_wheels = out_dir / f"{prefix}_povprecje_po_kolesih.png"
        plt.figure(figsize=(10, 4))
        plt.bar(self._wheel_names, wheel_means)
        plt.xlabel("kolo")
        plt.ylabel("povprecna normalna sila [N]")
        plt.title("Povprecna normalna sila po kolesih")
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(png_wheels, dpi=200)
        plt.close()

        # 6) Stevilo kontaktov skozi cas
        png_contact_count_time = out_dir / f"{prefix}_stevilo_kontaktov_skozi_cas.png"
        plt.figure(figsize=(10, 4))
        plt.plot(t, nL, label="Leva stran")
        plt.plot(t, nD, label="Desna stran")
        plt.plot(t, nT, label="Skupaj")
        plt.xlabel("cas voznje [s]")
        plt.ylabel("stevilo kontaktov [-]")
        plt.title("Stevilo kontaktov koles s terenom skozi cas")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_contact_count_time, dpi=200)

        # Roll in pitch sasije skozi cas - pomembno za E3 neraven teren
        png_roll_pitch = out_dir / f"{prefix}_roll_pitch_sasije.png"
        plt.figure(figsize=(10, 4))
        plt.plot(t, roll_deg, label="roll sasije")
        plt.plot(t, pitch_deg, label="pitch sasije")
        plt.xlabel("cas voznje [s]")
        plt.ylabel("kot [deg]")
        plt.title("Roll in pitch sasije skozi cas")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_roll_pitch, dpi=200)
        plt.close()
        plt.close()

        # 7) Skupno stevilo kontaktov po kolesih
        wheel_counts = []
        for wheel_name in self._wheel_names:
            key = f"n_{wheel_name}"
            vals = np.array([r[key] for r in rows], dtype=float)
            wheel_counts.append(float(vals.sum()))

        png_contact_count_wheels = out_dir / f"{prefix}_stevilo_kontaktov_po_kolesih.png"
        plt.figure(figsize=(10, 4))
        plt.bar(self._wheel_names, wheel_counts)
        plt.xlabel("kolo")
        plt.ylabel("skupno stevilo zaznanih kontaktov [-]")
        plt.title("Porazdelitev kontaktov po posameznih kolesih")
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(png_contact_count_wheels, dpi=200)
        plt.close()

        png_contact_xy = None
        png_contact_heatmap = None
        if point_rows:
            cx = np.array([r["contact_x"] for r in point_rows], dtype=float)
            cy = np.array([r["contact_y"] for r in point_rows], dtype=float)
            cf = np.array([r["normal_force_N"] for r in point_rows], dtype=float)
            sx = np.array([r["x"] for r in rows], dtype=float)
            sy = np.array([r["y"] for r in rows], dtype=float)

            # 8) XY zemljevid kontaktnih tock: kje nastajajo kontakti
            png_contact_xy = out_dir / f"{prefix}_kontaktne_tocke_xy.png"
            plt.figure(figsize=(6, 6))
            sc = plt.scatter(cx, cy, c=cf, s=12, alpha=0.75)
            plt.plot(sx, sy, linewidth=1.0, label="trajektorija sasije")
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.title("Lokacije kontaktnih tock na terenu")
            plt.grid(True)
            plt.axis("equal")
            plt.legend()
            cbar = plt.colorbar(sc)
            cbar.set_label("normalna sila kontakta [N]")
            plt.tight_layout()
            plt.savefig(png_contact_xy, dpi=200)
            plt.close()

            # 9) Gostota kontaktov v ravnini x-y
            png_contact_heatmap = out_dir / f"{prefix}_gostota_kontaktov_xy.png"
            plt.figure(figsize=(6, 6))
            hist = plt.hist2d(cx, cy, bins=35)
            plt.plot(sx, sy, linewidth=1.0, label="trajektorija sasije")
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.title("Gostota kontaktov na terenu")
            plt.grid(True)
            plt.axis("equal")
            plt.legend()
            cbar = plt.colorbar(hist[3])
            cbar.set_label("stevilo kontaktov v celici [-]")
            plt.tight_layout()
            plt.savefig(png_contact_heatmap, dpi=200)
            plt.close()

        # 10) Efektivni zdrs sigma_v in sigma_omega skozi cas
        png_slip = None
        png_slip_commands = None
        if slip_rows:
            ts = np.array([r["t"] for r in slip_rows], dtype=float)
            sv = np.array([r["sigma_v_eff"] for r in slip_rows], dtype=float)
            sw = np.array([r["sigma_omega_eff"] for r in slip_rows], dtype=float)
            vv_ideal = np.array([r["v_cmd_ideal"] for r in slip_rows], dtype=float)
            ww_ideal = np.array([r["omega_cmd_ideal"] for r in slip_rows], dtype=float)
            vv_eff = np.array([r["v_cmd_eff"] for r in slip_rows], dtype=float)
            ww_eff = np.array([r["omega_cmd_eff"] for r in slip_rows], dtype=float)

            png_slip = out_dir / f"{prefix}_efektivni_zdrs_sigma.png"
            plt.figure(figsize=(10, 4))
            plt.plot(ts, sv, label="sigma_v - vzdolzni/translacijski zdrs")
            plt.plot(ts, sw, label="sigma_omega - kotni/rotacijski zdrs")
            plt.ylim(-0.02, 1.02)
            plt.xlabel("cas voznje [s]")
            plt.ylabel("efektivni zdrs [-]")
            plt.title("Efektivni zdrs skozi cas")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(png_slip, dpi=200)
            plt.close()

            png_slip_commands = out_dir / f"{prefix}_idealni_in_efektivni_ukazi.png"
            plt.figure(figsize=(10, 5))
            plt.plot(ts, vv_ideal, label="v_ideal ukaz")
            plt.plot(ts, vv_eff, label="v_eff ukaz")
            plt.plot(ts, ww_ideal, label="omega_ideal ukaz")
            plt.plot(ts, ww_eff, label="omega_eff ukaz")
            plt.xlabel("cas voznje [s]")
            plt.ylabel("normaliziran ukaz [-]")
            plt.title("Idealni in efektivni del ukaza po upostevanju zdrsa")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(png_slip_commands, dpi=200)
            plt.close()

        # 11) Ocena moci in energije pogona
        png_drive_power = None
        png_drive_energy = None
        png_energy_actuators = None
        if energy_rows:
            te = np.array([r["t"] for r in energy_rows], dtype=float)
            p_signed = np.array([r["P_signed_total_W"] for r in energy_rows], dtype=float)
            p_pos = np.array([r["P_positive_total_W"] for r in energy_rows], dtype=float)
            p_abs = np.array([r["P_abs_total_W"] for r in energy_rows], dtype=float)
            p_left = np.array([r["P_positive_left_W"] for r in energy_rows], dtype=float)
            p_right = np.array([r["P_positive_right_W"] for r in energy_rows], dtype=float)
            e_pos = np.array([r["E_positive_total_J"] for r in energy_rows], dtype=float)
            e_abs = np.array([r["E_abs_total_J"] for r in energy_rows], dtype=float)

            png_drive_power = out_dir / f"{prefix}_moc_pogona.png"
            plt.figure(figsize=(10, 5))
            plt.plot(te, p_pos, label="pozitivna moc pogona - poraba")
            plt.plot(te, p_abs, label="absolutna moc pogona - napor")
            plt.plot(te, p_signed, label="predznacena mehanska moc")
            plt.plot(te, p_left, linestyle="--", label="pozitivna moc levo")
            plt.plot(te, p_right, linestyle="--", label="pozitivna moc desno")
            plt.xlabel("cas voznje [s]")
            plt.ylabel("moc [W]")
            plt.title("Ocena moci pogona skozi cas")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(png_drive_power, dpi=200)
            plt.close()

            png_drive_energy = out_dir / f"{prefix}_energija_pogona_integral.png"
            plt.figure(figsize=(10, 4))
            plt.plot(te, e_pos, label="E pozitivna = integral pozitivne moci")
            plt.plot(te, e_abs, label="E absolutna = integral absolutne moci")
            plt.xlabel("cas voznje [s]")
            plt.ylabel("energija [J]")
            plt.title("Ocena energije pogona - integral moci")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(png_drive_energy, dpi=200)
            plt.close()

            png_energy_actuators = out_dir / f"{prefix}_energija_po_motorjih.png"
            plt.figure(figsize=(10, 4))
            plt.bar(self._actuator_names, energy_per_actuator_positive_J)
            plt.xlabel("motor")
            plt.ylabel("pozitivna energija [J]")
            plt.title("Porazdelitev porabljene energije po motorjih")
            plt.grid(True, axis="y")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(png_energy_actuators, dpi=200)
            plt.close()

        total_contacts = int(sum(r["n_contacts_total"] for r in rows))
        total_contacts_left = int(sum(r["n_contacts_left"] for r in rows))
        total_contacts_right = int(sum(r["n_contacts_right"] for r in rows))

        print("\nShranjena analiza kontaktov:")
        print(f"  CSV sile/stevila: {csv_path}")
        if point_rows:
            print(f"  CSV tocke: {points_csv_path}")
        if slip_rows:
            print(f"  CSV zdrs: {slip_csv_path}")
        if energy_rows:
            print(f"  CSV energija: {energy_csv_path}")
            print(f"  Energija pogona E+ = {energy_positive_total_J:.3f} J = {energy_positive_total_J / 3600.0:.6f} Wh")
            print(f"  Energija pogona |E| = {energy_abs_total_J:.3f} J = {energy_abs_total_J / 3600.0:.6f} Wh")
        print(f"  Skupaj kontaktnih zapisov: {total_contacts}")
        print(f"  Levo: {total_contacts_left}, desno: {total_contacts_right}")
        print(f"  Graf 1: {png_forces}")
        print(f"  Graf 2: {png_balance}")
        print(f"  Graf 3: {png_wheels}")
        print(f"  Graf 4: {png_contact_count_time}")
        print(f"  Graf 5: {png_contact_count_wheels}")
        print(f"  Graf roll/pitch: {png_roll_pitch}")
        if png_contact_xy is not None:
            print(f"  Graf 6: {png_contact_xy}")
        if png_contact_heatmap is not None:
            print(f"  Graf 7: {png_contact_heatmap}")
        if png_slip is not None:
            print(f"  Graf 8: {png_slip}")
        if png_slip_commands is not None:
            print(f"  Graf 9: {png_slip_commands}")
        if png_drive_power is not None:
            print(f"  Graf 10: {png_drive_power}")
        if png_drive_energy is not None:
            print(f"  Graf 11: {png_drive_energy}")
        if png_energy_actuators is not None:
            print(f"  Graf 12: {png_energy_actuators}")
        print()

        return {
            "csv": str(csv_path),
            "points_csv": str(points_csv_path) if point_rows else None,
            "forces_png": str(png_forces),
            "balance_png": str(png_balance),
            "wheels_png": str(png_wheels),
            "count_time_png": str(png_contact_count_time),
            "count_wheels_png": str(png_contact_count_wheels),
            "contact_xy_png": str(png_contact_xy) if png_contact_xy is not None else None,
            "contact_heatmap_png": str(png_contact_heatmap) if png_contact_heatmap is not None else None,
            "slip_csv": str(slip_csv_path) if slip_rows else None,
            "slip_png": str(png_slip) if png_slip is not None else None,
            "slip_commands_png": str(png_slip_commands) if png_slip_commands is not None else None,
            "energy_csv": str(energy_csv_path) if energy_rows else None,
            "drive_power_png": str(png_drive_power) if png_drive_power is not None else None,
            "drive_energy_png": str(png_drive_energy) if png_drive_energy is not None else None,
            "energy_actuators_png": str(png_energy_actuators) if png_energy_actuators is not None else None,
            "energy_positive_total_J": energy_positive_total_J,
            "energy_abs_total_J": energy_abs_total_J,
            "energy_signed_total_J": energy_signed_total_J,
            "out_dir": str(out_dir),
            "total_contacts": total_contacts,
            "total_contacts_left": total_contacts_left,
            "total_contacts_right": total_contacts_right,
        }

    def _compute_stats(self):
        """Vrne statistiko premika in orientacije."""
        with self._lock:
            # Varno preberemo zacetni polozaj in zacetni yaw
            start_pos = None if self._start_pos is None else self._start_pos.copy()
            start_yaw = self._start_yaw

        # Ce voznja se ni bila zaceta, ni statistike
        if start_pos is None:
            return None

        with self._mj_lock:
            # Preberemo trenutni koncni polozaj in orientacijo sasije
            end_pos = self.data.body("sasija").xpos.copy()
            end_yaw = self._quat_to_yaw(self._qpos[self._root_qposadr + 3:self._root_qposadr + 7])

        # Razlika polozaja samo v ravnini x-y
        delta = end_pos[:2] - start_pos[:2]

        # Skupna prevozena razdalja med zacetkom in koncem
        dist = float(np.linalg.norm(delta))

        # Sprememba orientacije
        dyaw = self._wrap_angle(end_yaw - start_yaw)

        # Vrnemo vse uporabne podatke v slovarju
        return {
            "x0": float(start_pos[0]),
            "y0": float(start_pos[1]),
            "x1": float(end_pos[0]),
            "y1": float(end_pos[1]),
            "dx": float(delta[0]),
            "dy": float(delta[1]),
            "dist": dist,
            "yaw0_deg": float(np.degrees(start_yaw)),
            "yaw1_deg": float(np.degrees(end_yaw)),
            "dyaw_deg": float(np.degrees(dyaw)),
        }

    def _add_capsule_locked(self, p1: np.ndarray, p2: np.ndarray, radius: float = 0.02):
        """Izrise kapsulo med dvema tockama."""
        # Ce viewer ne obstaja, nimamo kam risati
        if self._viewer is None:
            return

        # user_scn je uporabniska scena, kamor lahko dodajamo svoje geometrije
        scn = self._viewer.user_scn

        # Ce je scena polna, ne moremo dodati nove geometrije
        if scn.ngeom >= scn.maxgeom:
            return

        # Vzamemo naslednji prost geom v sceni
        g = scn.geoms[scn.ngeom]

        # Inicializiramo geometrijo tipa kapsula
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.zeros(3, dtype=np.float64),
            pos=np.zeros(3, dtype=np.float64),
            mat=np.eye(3, dtype=np.float64).flatten(),
            rgba=self.TRAJ_RGBA,
        )

        # Kapsulo raztegnemo med p1 in p2, s podanim polmerom
        mujoco.mjv_connector(
            g,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            width=radius,
            from_=p1,
            to=p2,
        )

        # Povecamo stevilo narisanih geometrij
        scn.ngeom += 1

    def _redraw_trajectory_locked(self):
        """Ponovno narise celotno trajektorijo."""
        # Ce viewer ne obstaja, ne risemo nic
        if self._viewer is None:
            return

        with self._lock:
            # Naredimo lokalni kopiji trajektorije
            xs = list(self._traj_x)
            ys = list(self._traj_y)

        # Pocistimo prej narisano uporabnisko sceno
        self._viewer.user_scn.ngeom = 0

        # Trajektorijo dvignemo malo nad vozilo, da se lepse vidi
        z = float(self.data.body("sasija").xpos[2]) + 0.05

        # Med vsakim parom sosednjih tock narisemo kapsulo
        for i in range(1, len(xs)):
            p1 = np.array([xs[i - 1], ys[i - 1], z], dtype=np.float64)
            p2 = np.array([xs[i], ys[i], z], dtype=np.float64)
            self._add_capsule_locked(p1, p2)

    def _set_drive_locked(self, left: float, right: float):
        """Nastavi ukaz leve in desne strani."""
        # Prvih 6 aktuatorjev pripada levi strani
        self.data.ctrl[:6] = left

        # Drugih 6 aktuatorjev pripada desni strani
        self.data.ctrl[6:] = right

    def _start_drive(self, mode: str):
        """Inicializira novo voznjo."""
        print("Start mode:", mode)

        with self._mj_lock:
            # Zabelezimo cas zacetka voznje v simulacijskem casu
            start_time = self.data.time

            # Shranimo zacetni polozaj sasije
            start_pos = self.data.body("sasija").xpos.copy()

            # Shranimo zacetno orientacijo vozila
            start_yaw = self._quat_to_yaw(self._qpos[self._root_qposadr + 3:self._root_qposadr + 7])

            # Ce viewer obstaja, pobrisemo staro trajektorijo
            if self._viewer is not None:
                self._viewer.user_scn.ngeom = 0

        with self._lock:
            # Pocistimo staro trajektorijo
            self._traj_x.clear()
            self._traj_y.clear()

            # Nastavimo nov nacin voznje
            self._mode = mode

            # Shranimo zacetni cas
            self._drive_start_t = start_time

            # Shranimo zacetni polozaj in orientacijo
            self._start_pos = start_pos
            self._start_yaw = start_yaw

            # Trenutni prejsnji yaw zacnemo pri istem kotu
            self._prev_yaw = start_yaw

            # Ponastavimo skupni zasuk
            self._total_rot_rad = 0.0

            # Pocistimo stare kontaktne podatke za novo voznjo
            self._contact_log.clear()
            self._contact_point_log.clear()
            self._contact_sample_counter = 0
            self._slip_log.clear()
            self._slip_sample_counter = 0
            self._energy_log.clear()
            self._energy_sample_counter = 0
            self._energy_signed_J = 0.0
            self._energy_positive_J = 0.0
            self._energy_abs_J = 0.0
            self._energy_per_actuator_positive_J[:] = 0.0
            self._energy_per_actuator_abs_J[:] = 0.0
            self._last_terramech = None
            self._terra_core.reset()

    def _stop_drive(self):
        """Ustavi voznjo."""
        with self._lock:
            # Ugasnemo aktivni nacin voznje
            self._mode = self.MODE_NONE

        with self._mj_lock:
            # Vsem motorjem nastavimo nicelni ukaz in pocistimo G4 zunanje sile
            self._ctrl[:] = 0.0
            self._qfrc_applied[:] = 0.0

        # Izracunamo statistiko voznje
        stats = self._compute_stats()

        # Shrani CSV + grafe kontaktnih sil za zadnjo voznjo
        analysis_files = self._save_contact_analysis()
        if stats is not None and analysis_files is not None:
            stats["analysis_dir"] = analysis_files["out_dir"]
            stats["total_contacts"] = analysis_files.get("total_contacts")
            stats["total_contacts_left"] = analysis_files.get("total_contacts_left")
            stats["total_contacts_right"] = analysis_files.get("total_contacts_right")
            stats["energy_positive_total_J"] = analysis_files.get("energy_positive_total_J", 0.0)
            stats["energy_positive_total_Wh"] = analysis_files.get("energy_positive_total_J", 0.0) / 3600.0

        # Ce GUI obstaja, odpremo okno s statistiko
        if self._root is not None:
           with self._lock:
              self._pending_stats = stats
              self._show_stats_requested = True

    def _reset(self):
        """Ponastavi simulacijo."""
        with self._lock:
            # Pocistimo stanje voznje
            self._mode = self.MODE_NONE
            self._drive_start_t = 0.0
            self._traj_x.clear()
            self._traj_y.clear()
            self._start_pos = None
            self._start_yaw = 0.0
            self._prev_yaw = 0.0
            self._total_rot_rad = 0.0
            self._contact_log.clear()
            self._contact_point_log.clear()
            self._contact_sample_counter = 0
            self._slip_log.clear()
            self._slip_sample_counter = 0
            self._energy_log.clear()
            self._energy_sample_counter = 0
            self._energy_signed_J = 0.0
            self._energy_positive_J = 0.0
            self._energy_abs_J = 0.0
            self._energy_per_actuator_positive_J[:] = 0.0
            self._energy_per_actuator_abs_J[:] = 0.0
            self._last_terramech = None
            self._terra_core.reset()

        with self._mj_lock:
            # MuJoCo resetira data na zacetno stanje iz modela
            mujoco.mj_resetData(self.model, self.data)

            # Pocistimo tudi zunanje G4 sile.
            self._qfrc_applied[:] = 0.0

            # Izvede forward izracun, da se stanja in polozaji pravilno osvezijo
            mujoco.mj_forward(self.model, self.data)

            # Ce viewer obstaja, pobrisemo narisano trajektorijo
            if self._viewer is not None:
                self._viewer.user_scn.ngeom = 0

        # Ponastavimo informacijo o zadnjem materialu
        self._last_material_id = None

    def _physics_loop(self):
        # Casovni korak simulacije iz modela
        dt = self.model.opt.timestep

        # Glavna zanka fizike tece, dokler je zastavica vklopljena
        while self._physics_running:
            # Realni zacetni cas trenutne iteracije
            t_start = time.perf_counter()

            with self._lock:
                # Preberemo trenutno stanje voznje
                mode = self._mode
                drive_start_t = self._drive_start_t
                prev_yaw = self._prev_yaw

            # Zastavici za kasnejsi reset ali stop
            should_stop = False
            should_reset = False

            with self._mj_lock:
                # Koliko simulacijskega casa je minilo od zacetka voznje
                elapsed = self.data.time - drive_start_t if mode is not None else 0.0

                # Samodejni stop po pretecenem trajanju voznje
                if mode is not None and elapsed > self._drive_duration:
                    should_stop = True
                    mode = self.MODE_NONE

                # Trenutna pozicija sasije
                pos = self.data.body("sasija").xpos
                x = float(pos[0])
                y = float(pos[1])

                # Izberemo parametre podlage glede na polozaj
                mu, lam, mat_id, mat_name = self.terrain_params(x, y)

                # --- G4: ukaz -> vcmd/omega_cmd -> FL/FR + upori ---
                # Gumbi se vedno dolocijo kot stari levi/desni ukaz, vendar jih
                # predamo G4 jedru kot zahtevano linearno in kotno hitrost.
                drive_info = None
                traction_scale = 0.0

                # Glede na aktivni nacin voznje dolocimo idealne leve/desne ukaze.
                if mode == self.MODE_STRAIGHT:
                    left_cmd = 1.0
                    right_cmd = 1.0

                elif mode == self.MODE_PIVOT:
                    # Leva stran nazaj, desna naprej -> vrtenje na mestu.
                    # Enaka vrednost kot v vmesnem porocilu.
                    left_cmd = -0.8
                    right_cmd = 0.8

                elif mode == self.MODE_CIRCLE:
                    # Desna stran hitreje od leve -> krog.
                    # Enaki vrednosti kot v vmesnem porocilu; hitrost skale nastavlja _max_track_speed_mps.
                    left_cmd = 0.6
                    right_cmd = 1.2

                elif mode == self.MODE_FIGURE8:
                    # Osmica: prvi del en zavoj, drugi del nasprotni zavoj.
                    # Enaki vrednosti in perioda kot v vmesnem porocilu.
                    period = 17.5
                    phase = elapsed % (2.0 * period)

                    if phase < period:
                        left_cmd = 0.8
                        right_cmd = 0.3
                    else:
                        left_cmd = 0.3
                        right_cmd = 0.8

                else:
                    left_cmd = 0.0
                    right_cmd = 0.0

                if mode is not None:
                    cmd_vel = self._left_right_command_to_cmd_vel(left_cmd, right_cmd)
                    loads = self._compute_contact_loads_locked()
                    state = self._body_planar_state_locked()

                    # Brez yaw-hold stabilizacije:
                    # - NARAVNOST ima left=1.0, right=1.0 -> omega_cmd = 0.0,
                    # - G4 ne dodaja korekcije orientacije.
                    # Tako se v trajektoriji vidi dejanski vpliv terena, kontaktov,
                    # normalnih sil, zdrsa in uporov.

                    terra = self._terra_core.step(
                        dt,
                        cmd=cmd_vel,
                        terrain={"mu": mu, "lambda": lam, "material": mat_name},
                        loads=loads,
                        state=state
                    )

                    b = max(0.05, float(self._track_width_m))
                    left_eff = terra["v_ref"] - 0.5 * b * terra["omega_ref"]
                    right_eff = terra["v_ref"] + 0.5 * b * terra["omega_ref"]

                    drive_info = dict(terra)
                    drive_info.update({
                        # Stari kljuci ostanejo zaradi obstojecega logiranja/grafov.
                        "left_eff": float(left_eff),
                        "right_eff": float(right_eff),
                        "v_cmd_ideal": float(cmd_vel["v_cmd"]),
                        "omega_cmd_ideal": float(cmd_vel["omega_cmd"]),
                        "v_cmd_eff": float(terra["v_ref"]),
                        "omega_cmd_eff": float(terra["omega_ref"]),
                        "sigma_v_eff": float(terra["sigma_v"]),
                        "sigma_omega_eff": float(terra["sigma_omega"]),
                        "drive_gain_mu": float(mu),
                        "traction_scale": float(terra["traction_scale"]),
                        "left_cmd_raw": float(left_cmd),
                        "right_cmd_raw": float(right_cmd),
                    })

                    traction_scale = drive_info["traction_scale"]

                    # Pogon je zdaj G4 force-injection, zato motornih ctrl ukazov ne uporabljamo.
                    self._ctrl[:] = 0.0
                    self._apply_terramechanics_forces_locked(drive_info)
                    self._last_terramech = drive_info

                else:
                    # Ce ni aktivnega nacina, vse motorje in zunanje sile ustavimo.
                    self._ctrl[:] = 0.0
                    self._qfrc_applied[:] = 0.0
                    self._last_terramech = None

                # Izpis samo, ko se material spremeni
                if mat_id != self._last_material_id:
                    if drive_info is not None:
                        print(
                            f"x={x:.2f}, y={y:.2f}, material={mat_name}, "
                            f"sigma_v={drive_info['sigma_v_eff']:.2f}, "
                            f"sigma_omega={drive_info['sigma_omega_eff']:.2f}, "
                            f"scale={traction_scale:.2f}"
                        )
                    else:
                        print(f"x={x:.2f}, y={y:.2f}, material={mat_name}")
                    self._last_material_id = mat_id
                # Redek debug izpis
                if self._traj_draw_counter % 200 == 0:
                    if drive_info is not None:
                        print(
                            f"x={x:.2f}, mat={mat_name}, "
                            f"sigma_v={drive_info['sigma_v_eff']:.2f}, "
                            f"sigma_omega={drive_info['sigma_omega_eff']:.2f}, "
                            f"scale={traction_scale:.2f}"
                        )

                # Izvedemo en fizikalni korak simulacije
                mujoco.mj_step(self.model, self.data)

                # Po koraku ponovno preberemo polozaj
                pos = self.data.body("sasija").xpos
                x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

                # Kontaktne sile, zdrs in energijo vzorcimo po mj_step,
                # ko je stanje simulacije posodobljeno.
                self._sample_contacts_locked(mode, mat_name, traction_scale)
                self._sample_slip_locked(mode, mat_name, mu, lam, drive_info)
                self._sample_energy_locked(mode, mat_name, drive_info, dt)

                # Ce gre tank predalec ali pade prenizko, naredimo reset
                if x < -30 or x > 30 or y < -30 or y > 30 or z < -0.2:
                    print("Reset: izven dovoljenega obmocja")
                    should_reset = True

                if not should_reset:
                    # Izracunamo nov yaw
                    current_yaw = self._quat_to_yaw(self._qpos[self._root_qposadr + 3:self._root_qposadr + 7])

                    # Sprememba yawa od prejsnjega koraka
                    dyaw = self._wrap_angle(current_yaw - prev_yaw)

                    with self._lock:
                        # Posodobimo shranjeni yaw
                        self._prev_yaw = current_yaw

                        # Pristejemo skupni zasuk
                        self._total_rot_rad += dyaw

                        # Posodobimo mode v skupnem stanju
                        self._mode = mode

                        # Ce aktivno vozimo, dodajamo tocke trajektorije
                        if mode is not None:
                            pos = self.data.body("sasija").xpos
                            x = float(pos[0])
                            y = float(pos[1])

                            # Dodamo novo tocko le, ce smo se dovolj premaknili
                            if (
                                len(self._traj_x) == 0
                                or np.hypot(x - self._traj_x[-1], y - self._traj_y[-1]) > 0.05
                            ):
                                self._traj_x.append(x)
                                self._traj_y.append(y)

                # Obcasno osvezimo risanje trajektorije
                if mode is not None:
                    self._traj_draw_counter += 1
                    if self._traj_draw_counter % 5 == 0:
                        self._redraw_trajectory_locked()
                else:
                    # Ce ni voznje, pobrisemo uporabnisko sceno
                    if self._viewer is not None:
                        self._viewer.user_scn.ngeom = 0

            # Reset naredimo izven MuJoCo zaklenjenega bloka
            if should_reset:
                self._reset()
                continue

            # Stop naredimo izven MuJoCo zaklenjenega bloka
            if should_stop:
                self._stop_drive()

            # Sinhronizacija realnega casa s simulacijskim korakom
            deadline = t_start + dt
            while time.perf_counter() < deadline:
                time.sleep(0)

    def _terrain_height_at(self, x: float, y: float) -> float:
        """
        Priblizna visina terena pri koordinatah (x, y).

        Uporabi MuJoCo hfield podatke, ce obstaja geom "terrain". Za raven teren
        uporabi zgornjo ploskev geoma "povrsina". Namen ni natancna kolizijska
        poizvedba, ampak varen zacetni z pri prestavljanju tanka.
        """
        x = float(x)
        y = float(y)

        # Neraven hfield teren.
        try:
            terrain_geom_id = self.model.geom("terrain").id
            hfield_id = int(self.model.geom_dataid[terrain_geom_id])
            if hfield_id >= 0:
                size = np.asarray(self.model.hfield_size[hfield_id], dtype=float)
                rx, ry, zscale, _base = [float(v) for v in size]
                nrow = int(self.model.hfield_nrow[hfield_id])
                ncol = int(self.model.hfield_ncol[hfield_id])
                adr = int(self.model.hfield_adr[hfield_id])
                hdata = np.asarray(
                    self.model.hfield_data[adr: adr + nrow * ncol],
                    dtype=float
                ).reshape(nrow, ncol)

                geom_pos = np.asarray(self.model.geom_pos[terrain_geom_id], dtype=float)
                lx = x - float(geom_pos[0])
                ly = y - float(geom_pos[1])

                u = (lx / rx + 1.0) * 0.5
                v = (ly / ry + 1.0) * 0.5
                u = max(0.0, min(1.0, u))
                v = max(0.0, min(1.0, v))

                col = int(round(u * (ncol - 1)))
                # Y os slike je obrnjena glede na svetovni y, podobno kot pri material_map.
                row = int(round((1.0 - v) * (nrow - 1)))
                row = max(0, min(nrow - 1, row))
                col = max(0, min(ncol - 1, col))

                return float(geom_pos[2] + hdata[row, col] * zscale)
        except Exception:
            pass

        # Raven box teren.
        try:
            flat_geom_id = self.model.geom("povrsina").id
            geom_pos = np.asarray(self.model.geom_pos[flat_geom_id], dtype=float)
            geom_size = np.asarray(self.model.geom_size[flat_geom_id], dtype=float)
            return float(geom_pos[2] + geom_size[2])
        except Exception:
            pass

        return 0.0

    def _teleport_tank_to_xy(self, x: float, y: float):
        """Postavi tank na novo koordinato (x, y) in ustavi hitrosti/pogon."""
        x = max(self._terrain_world_xmin, min(self._terrain_world_xmax, float(x)))
        y = max(self._terrain_world_ymin, min(self._terrain_world_ymax, float(y)))
        z = self._terrain_height_at(x, y) + self._placement_z_clearance

        with self._lock:
            # Prestavljanje obravnavamo kot novo zacetno stanje, ne kot nadaljevanje voznje.
            self._mode = self.MODE_NONE
            self._drive_start_t = 0.0
            self._start_pos = None
            self._start_yaw = 0.0
            self._prev_yaw = 0.0
            self._total_rot_rad = 0.0
            self._traj_x.clear()
            self._traj_y.clear()

        with self._mj_lock:
            qadr = self._root_qposadr

            # Ohrani trenutno orientacijo, spremeni samo polozaj free jointa.
            self._qpos[qadr + 0] = x
            self._qpos[qadr + 1] = y
            self._qpos[qadr + 2] = z

            # Poskrbi, da kvaternion ostane normiran.
            quat = np.asarray(self._qpos[qadr + 3: qadr + 7], dtype=float)
            qnorm = float(np.linalg.norm(quat))
            if qnorm > 1e-12:
                self._qpos[qadr + 3: qadr + 7] = quat / qnorm

            # Po prestavljanju ustavimo hitrosti, da tank ne "odleti" naprej.
            self._qvel[:] = 0.0
            self._ctrl[:] = 0.0
            if hasattr(self.data, "qacc_warmstart"):
                self.data.qacc_warmstart[:] = 0.0

            mujoco.mj_forward(self.model, self.data)

            if self._viewer is not None:
                self._viewer.user_scn.ngeom = 0

        self._last_material_id = None
        print(f"Tank prestavljen na x={x:.2f}, y={y:.2f}, z={z:.2f}")
        self._update_placement_canvas(force=True)

    def _world_to_canvas(self, x: float, y: float):
        """Pretvori svetovne koordinate v piksle na top-down zemljevidu."""
        size = self._placement_canvas_size
        u = (float(x) - self._terrain_world_xmin) / (self._terrain_world_xmax - self._terrain_world_xmin)
        v = (float(y) - self._terrain_world_ymin) / (self._terrain_world_ymax - self._terrain_world_ymin)
        px = u * size
        py = (1.0 - v) * size
        return float(px), float(py)

    def _canvas_to_world(self, px: float, py: float):
        """Pretvori klik na top-down zemljevidu v svetovne koordinate."""
        size = self._placement_canvas_size
        px = max(0.0, min(float(size), float(px)))
        py = max(0.0, min(float(size), float(py)))
        u = px / size
        v = 1.0 - py / size
        x = self._terrain_world_xmin + u * (self._terrain_world_xmax - self._terrain_world_xmin)
        y = self._terrain_world_ymin + v * (self._terrain_world_ymax - self._terrain_world_ymin)
        return float(x), float(y)

    def _current_tank_xy_yaw(self):
        """Vrne trenutni x, y in yaw tanka."""
        with self._mj_lock:
            qadr = self._root_qposadr
            x = float(self._qpos[qadr + 0])
            y = float(self._qpos[qadr + 1])
            yaw = self._quat_to_yaw(self._qpos[qadr + 3: qadr + 7])
        return x, y, yaw

    def _on_placement_window_close(self):
        self._placement_dragging = False
        self._placement_preview_xy = None
        if self._placement_window is not None:
            try:
                self._placement_window.destroy()
            except tk.TclError:
                pass
        self._placement_window = None
        self._placement_canvas = None
        self._placement_status_var = None
        self._placement_map_photo = None

    def _open_placement_window(self):
        """Odpre top-down okno, kjer lahko z misko primemo in prestavimo tank."""
        if self._root is None:
            return

        if self._placement_window is not None:
            try:
                self._placement_window.lift()
                self._placement_window.focus_force()
                return
            except tk.TclError:
                self._placement_window = None

        win = tk.Toplevel(self._root)
        self._placement_window = win
        win.title("Prestavljanje tanka po terenu")
        win.protocol("WM_DELETE_WINDOW", self._on_placement_window_close)

        info = (
            "Klikni na oznako tanka, drzi levo tipko in ga povleci na novo mesto. "
            "Ko tipko spustis, se tank prestavi v MuJoCo simulaciji."
        )
        tk.Label(win, text=info, wraplength=500, justify="left").pack(padx=10, pady=(10, 4))

        self._placement_status_var = tk.StringVar(value="Modra oznaka = trenutni polozaj tanka.")
        tk.Label(win, textvariable=self._placement_status_var, justify="left").pack(padx=10, pady=(0, 6))

        size = self._placement_canvas_size
        canvas = tk.Canvas(win, width=size, height=size, bg="white", highlightthickness=1, highlightbackground="black")
        canvas.pack(padx=10, pady=8)
        self._placement_canvas = canvas

        canvas.bind("<ButtonPress-1>", self._placement_on_press)
        canvas.bind("<B1-Motion>", self._placement_on_motion)
        canvas.bind("<ButtonRelease-1>", self._placement_on_release)

        btn_frame = tk.Frame(win)
        btn_frame.pack(padx=10, pady=(0, 10), fill="x")

        tk.Button(
            btn_frame,
            text="Osvezi zemljevid",
            command=lambda: self._update_placement_canvas(force=True)
        ).pack(side="left", padx=3)

        tk.Button(
            btn_frame,
            text="Zapri",
            command=self._on_placement_window_close
        ).pack(side="right", padx=3)

        self._update_placement_canvas(force=True)

    def _update_placement_canvas(self, force: bool = False):
        """Ponovno narise top-down zemljevid in oznako tanka."""
        if self._placement_canvas is None or self._placement_window is None:
            return

        try:
            if not self._placement_window.winfo_exists():
                return
        except tk.TclError:
            return

        # Med vlecenjem ne prepisujemo preview oznake z dejanskim polozajem.
        if self._placement_dragging and not force:
            return

        canvas = self._placement_canvas
        size = self._placement_canvas_size
        canvas.delete("all")

        # Ozadje: vizualna materialna mapa, ce obstaja. Ce ne, narisemo samo mrezo.
        try:
            base = Path(__file__).resolve().parent
            vis_path = base / "terrains" / "material_map_vis.png"
            if vis_path.exists():
                img = Image.open(str(vis_path)).convert("RGB").resize((size, size), Image.Resampling.NEAREST)
                self._placement_map_photo = ImageTk.PhotoImage(img)
                canvas.create_image(0, 0, anchor="nw", image=self._placement_map_photo)
            else:
                canvas.create_rectangle(0, 0, size, size, fill="white", outline="")
        except Exception:
            canvas.create_rectangle(0, 0, size, size, fill="white", outline="")

        # Mreza in osi.
        for value in range(-30, 31, 10):
            px1, py1 = self._world_to_canvas(value, self._terrain_world_ymin)
            px2, py2 = self._world_to_canvas(value, self._terrain_world_ymax)
            canvas.create_line(px1, py1, px2, py2, fill="#777777")
            px1, py1 = self._world_to_canvas(self._terrain_world_xmin, value)
            px2, py2 = self._world_to_canvas(self._terrain_world_xmax, value)
            canvas.create_line(px1, py1, px2, py2, fill="#777777")

        canvas.create_text(8, 8, anchor="nw", text="x/y pogled od zgoraj", fill="black")
        canvas.create_text(size - 8, 8, anchor="ne", text="+y", fill="black")
        canvas.create_text(size - 8, size - 8, anchor="se", text="+x proti desni", fill="black")

        # Polozaj tanka ali preview med vlecenjem.
        if self._placement_preview_xy is not None:
            x, y = self._placement_preview_xy
            yaw = 0.0
            marker_fill = "orange"
            label = f"novo: x={x:.2f}, y={y:.2f}"
        else:
            x, y, yaw = self._current_tank_xy_yaw()
            marker_fill = "cyan"
            label = f"tank: x={x:.2f}, y={y:.2f}"

        px, py = self._world_to_canvas(x, y)
        r = 11
        canvas.create_oval(px - r, py - r, px + r, py + r, fill=marker_fill, outline="black", width=2)

        # Smer vozila.
        arrow_len = 26
        ax = px + arrow_len * np.cos(yaw)
        ay = py - arrow_len * np.sin(yaw)
        canvas.create_line(px, py, ax, ay, arrow="last", width=2)
        canvas.create_text(px + 14, py - 18, text=label, anchor="w", fill="black")

    def _placement_on_press(self, event):
        if self._placement_canvas is None:
            return

        tank_x, tank_y, _yaw = self._current_tank_xy_yaw()
        tank_px, tank_py = self._world_to_canvas(tank_x, tank_y)
        dist_px = float(np.hypot(float(event.x) - tank_px, float(event.y) - tank_py))

        if dist_px <= 22.0:
            self._placement_dragging = True
            self._placement_preview_xy = (tank_x, tank_y)
            if self._placement_status_var is not None:
                self._placement_status_var.set("Tank je prijet. Povleci ga na novo mesto in spusti tipko.")
            self._update_placement_canvas(force=True)
        else:
            if self._placement_status_var is not None:
                x, y = self._canvas_to_world(event.x, event.y)
                self._placement_status_var.set(
                    f"Najprej klikni na oznako tanka. Kliknjeno mesto: x={x:.2f}, y={y:.2f}."
                )

    def _placement_on_motion(self, event):
        if not self._placement_dragging:
            return
        x, y = self._canvas_to_world(event.x, event.y)
        self._placement_preview_xy = (x, y)
        if self._placement_status_var is not None:
            self._placement_status_var.set(f"Novo mesto: x={x:.2f}, y={y:.2f}. Spusti tipko za prestavitev.")
        self._update_placement_canvas(force=True)

    def _placement_on_release(self, event):
        if not self._placement_dragging:
            return
        x, y = self._canvas_to_world(event.x, event.y)
        self._placement_dragging = False
        self._placement_preview_xy = None
        self._teleport_tank_to_xy(x, y)
        if self._placement_status_var is not None:
            self._placement_status_var.set(f"Tank prestavljen na x={x:.2f}, y={y:.2f}.")
        self._update_placement_canvas(force=True)

    def _build_gui(self):
        """Zgradi kontrolno okno."""
        # Ustvarimo glavno tkinter okno
        self._root = tk.Tk()

        # Naslov okna
        self._root.title("Upravljanje tanka")

        # Skupne nastavitve za gumbe
        btn_cfg = {"width": 20}

        # Gumb za reset simulacije
        tk.Button(
            self._root,
            text="PONASTAVITEV",
            command=self._reset,
            bg="lightblue",
            **btn_cfg
        ).pack(pady=4)

        # Gumb odpre top-down zemljevid, kjer lahko tank primemo z misko
        # in ga spustimo na poljubno mesto na terenu.
        tk.Button(
            self._root,
            text="PRESTAVI TANK Z MISKO",
            command=self._open_placement_window,
            bg="khaki",
            **btn_cfg
        ).pack(pady=4)

        # Gumb za voznjo naravnost
        tk.Button(
            self._root,
            text="NARAVNOST",
            command=lambda: self._start_drive(self.MODE_STRAIGHT),
            **btn_cfg
        ).pack(pady=4)

        # Gumb za pivot
        tk.Button(
            self._root,
            text="PIVOT",
            command=lambda: self._start_drive(self.MODE_PIVOT),
            **btn_cfg
        ).pack(pady=4)

        # Gumb za krog
        tk.Button(
            self._root,
            text="KROG",
            command=lambda: self._start_drive(self.MODE_CIRCLE),
            **btn_cfg
        ).pack(pady=4)

        # Gumb za osmico
        tk.Button(
            self._root,
            text="OSMICA",
            command=lambda: self._start_drive(self.MODE_FIGURE8),
            **btn_cfg
        ).pack(pady=4)

        # Gumb za ustavitev
        tk.Button(
            self._root,
            text="STOP",
            command=self._stop_drive,
            bg="red",
            fg="white",
            **btn_cfg
        ).pack(pady=8)

    def _show_stats_window(self, stats):
        """Prikaze rezultate voznje."""
        # Ce ni statistike, ne odpremo okna
        if stats is None:
            return

        # Novo manjse okno nad glavnim GUI
        win = tk.Toplevel(self._root)

        # Naslov okna
        win.title("Rezultati voznje")

        # Besedilo s podatki o voznji
        text = (
            f"Zacetek:\n"
            f"  x0 = {stats['x0']:.3f} m\n"
            f"  y0 = {stats['y0']:.3f} m\n\n"
            f"Konec:\n"
            f"  x1 = {stats['x1']:.3f} m\n"
            f"  y1 = {stats['y1']:.3f} m\n\n"
            f"Premik:\n"
            f"  dx      = {stats['dx']:.3f} m\n"
            f"  dy      = {stats['dy']:.3f} m\n"
            f"  skupaj  = {stats['dist']:.3f} m\n\n"
            f"Orientacija:\n"
            f"  yaw0 = {stats['yaw0_deg']:.2f} st\n"
            f"  yaw1 = {stats['yaw1_deg']:.2f} st\n"
            f"  dyaw = {stats['dyaw_deg']:.2f} st"
        )

        if "analysis_dir" in stats:
            text += f"\n\nKontakti:\n"
            text += f"  skupaj = {stats.get('total_contacts', 0)}\n"
            text += f"  levo   = {stats.get('total_contacts_left', 0)}\n"
            text += f"  desno  = {stats.get('total_contacts_right', 0)}\n"
            text += f"\nEnergija pogona:\n"
            text += f"  E+ = {stats.get('energy_positive_total_J', 0.0):.3f} J\n"
            text += f"     = {stats.get('energy_positive_total_Wh', 0.0):.6f} Wh\n"
            text += f"\nGrafi in CSV:\n  {stats['analysis_dir']}"

        # Label z rezultati
        tk.Label(
            win,
            text=text,
            justify="left",
            font=("Arial", 11)
        ).pack(padx=15, pady=15)

    def run(self):
        """Zazene GUI, fiziko in viewer."""
        # Najprej zgradimo kontrolni GUI
        self._build_gui()

        try:
            # Odpremo MuJoCo viewer v pasivnem nacinu
            with viewer.launch_passive(self.model, self.data) as v:
                # Shranimo viewer objekt
                self._viewer = v

                # Oznacimo, da physics loop tece
                self._physics_running = True

                # Ustvarimo nit za fiziko
                self._physics_thread = threading.Thread(
                    target=self._physics_loop,
                    daemon=False
                )

                # Zazenemo physics nit
                self._physics_thread.start()

                # Glavna zanka viewerja in GUI
                while v.is_running():
                    try:
                        # Osvezi tkinter dogodke in izgled
                        self._root.update_idletasks()
                        self._root.update()
                        with self._lock:
                            if self._show_stats_requested:
                                stats = self._pending_stats
                                self._pending_stats = None
                                self._show_stats_requested = False
                            else:
                                stats = None

                        if stats is not None:
                            self._show_stats_window(stats)

                        # Ce je odprto okno za prestavljanje, osvezimo marker tanka
                        # priblizno 10-krat na sekundo.
                        now_for_placement = time.perf_counter()
                        if now_for_placement - self._placement_last_redraw > 0.10:
                            self._placement_last_redraw = now_for_placement
                            self._update_placement_canvas()
                    except tk.TclError:
                        # Ce se GUI okno zapre, prekinemo zanko
                        break

                    with self._mj_lock:
                        # Uskladi viewer s trenutnim stanjem simulacije
                        v.sync()

                    # Kratek sleep, da GUI ne porabi prevec CPU
                    time.sleep(0.005)

        finally:
            # Ob izhodu ustavimo physics loop
            self._physics_running = False


if __name__ == "__main__":
    # Mapa, kjer lezi trenutna Python datoteka
    base = Path(__file__).resolve().parent

    # Pot do XML modela. Popravljena verzija uporablja XML s SITE aktuatorji.
    xml_path = base / "E3_neraven_asfalt.xml"
    #if not xml_path.exists():
        # Fallback, ce datoteko preimenujes nazaj v Neraven_teren.xml.
       #xml_path = base / "Neraven_teren.xml"

    # Ustvarimo simulator
    sim = TankSimulator(str(xml_path))

    # Zazenemo simulator
    sim.run()