******Tracked vehicle-skid-steer-project******

Projekt implementira simulacijo skid-steer vozila oziroma goseničarja v okolju MuJoCo. Model vključuje raven in neraven teren, tri tipe podlage, terramechanics/slip model, analizo kontaktov s podlago ter ROS 2 bridge.

*****Glavne datoteke:

- `Koncna_faza.py` – glavna simulacija v MuJoCo, GUI, terramechanics/slip core, analiza kontaktov in shranjevanje grafov
- `Koncna_faza_E3.py` – verzija za E3, kjer se dodatno shranjuje roll/pitch šasije
- `ROS_bridge.py` – ROS 2 bridge za upravljanje prek `/cmd_vel`
- `Raven_teren_SITE.xml` – raven teren
- `Neraven_teren_SITE.xml` – neraven teren
- `E1_asfalt.xml` – scenarij za asfalt
- `E1_blato.xml` – scenarij za blato/zemljo
- `E1_pesek.xml` – scenarij za pesek
- `E2_prehod_materialov.xml` – scenarij prehoda med materiali
- `E3_neraven_asfalt.xml` – scenarij za neraven teren
- `material_map_SITE.py` – generiranje materialne mape
- `naredi_slika.py` – generiranje vizualne materialne mape
- `naredi_teren.py` – generiranje heightmap terena

****Mape:

- `terrains/` – materialne mape in heightmap slike
- `eksperimenti/` – rezultati eksperimentov, grafi in CSV datoteke
- `eksperimenti/E1_materiali/` – rezultati za vpliv tipa podlage, potem pa posebi za vsak del BLATO, ASFALT IN PESEK imamo posebne grafe razlicnih manevr
- `eksperimenti/E2/` – rezultati za prehod med materiali
- `eksperimenti/E3/` – rezultati za neraven teren
- `bags/` – rosbag2 posnetki


********Namestitev**********

Uporabljeno okolje:

- Ubuntu Linux
- Python 3
- MuJoCo Python API
- NumPy
- Matplotlib
- Pillow
- ROS 2 Humble

Namestitev Python okolja:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install mujoco numpy matplotlib pillow
```

Če uporabljamo `requirements.txt`:

```bash
pip install -r requirements.txt
```

******Generiranje terena in materialnih map******

Logična materialna mapa:

```bash
python material_map_SITE.py
```

Vizualna materialna mapa:

```bash
python naredi_slika.py
```

Heightmap za neraven teren:

```bash
python naredi_teren.py
```

****Zagon simulacije****

V datoteki `Koncna_faza.py` se na dnu izbere XML model, na primer:

```python
xml_path = base / "E1_asfalt.xml"
```

-Zagon:

```bash
source .venv/bin/activate
python Koncna_faza.py
```

Po zagonu se odpre MuJoCo viewer in GUI z gumbi:

- `NARAVNOST`
- `PIVOT`
- `KROG`
- `OSMICA`
- `STOP`
- `PONASTAVITEV`

Pomembno: rezultati in grafi se shranijo šele, ko se vožnja zaključi z gumbom `STOP`.
Rezultati se shranjujejo v mapo: eksperimenti,shranimo CSV datoteke in grafe:
                    - normalne sile levo/desno
                    - razlika normalnih sil levo/desno
                    - povprečna normalna sila po kolesih
                    - število kontaktov skozi čas
                    - porazdelitev kontaktov po kolesih
                    - kontaktne točke v ravnini x-y
                    - gostota kontaktov na terenu
                    - efektivni zdrs
                    - idealni in efektivni ukazi
                    - moč pogona
                    - energija pogona
                    - energija po motorjih
                    - roll/pitch šasije pri E

*****ROS 2 bridge******

Zagon ROS 2 bridge:

```bash
source /opt/ros/humble/setup.bash
source .venv/bin/activate
python ROS_bridge.py
```

Primer ukaza za vožnjo naprej:

```bash
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

Stop ukaz:

```bash
ros2 topic pub -1 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```


