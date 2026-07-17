# Linux Machine Runtime Setup Guide

คู่มือนี้ใช้สำหรับผู้ใช้งานระบบบน `Linux / Jetson` เพื่อเตรียมโครงสร้างไฟล์ของโปรแกรมให้ถูกต้อง สร้าง `shell script` สำหรับเรียกไฟล์หลัก `index.py` และเตรียมภาพประกอบแต่ละขั้นตอนเพื่อนำไปเรียบเรียงต่อในเอกสาร Word

---

## 1. วัตถุประสงค์ของคู่มือ

คู่มือนี้ครอบคลุมงานต่อไปนี้:

1. ดึงโค้ดจาก GitHub ลงเครื่อง Linux/Jetson
2. จัดวางโครงสร้างไฟล์และโฟลเดอร์ของโปรแกรมให้ถูกต้อง
3. ตรวจสอบว่าไฟล์หลักและไฟล์ประกอบอยู่ครบตามพาธที่โปรแกรมต้องใช้
4. สร้าง `shell script` สำหรับเรียกโปรแกรมหลัก
5. ตั้งสิทธิ์ให้สคริปต์สามารถรันได้
6. ทดลองรันโปรแกรม
7. บันทึกภาพหน้าจอของแต่ละขั้นตอนเพื่อนำไปอธิบายต่อในเอกสาร Word

---

## 2. โครงสร้างโฟลเดอร์มาตรฐาน

ให้จัดวางโครงสร้างโปรแกรมในเครื่อง Linux/Jetson ตามตัวอย่างด้านล่าง

```text
AI-Vision-Wave/
├── index.py
├── calibration_data.json
├── offsets.json
├── place_targets.json
├── yolov26s_fixed.pt
├── as7265x_sparkfun_python.py
├── dashboard_app.py
├── run_dashboard_sync.py
├── requirements-dashboard.txt
├── .env.customer.example
├── Model/
│   ├── CatBoost.joblib
│   ├── scaler.joblib
│   ├── label_encoder.joblib
│   └── used_features.joblib
├── modules/
│   ├── __init__.py
│   ├── camera.py
│   ├── config.py
│   ├── database.py
│   ├── detector.py
│   ├── robot.py
│   ├── spectrum.py
│   ├── tracker.py
│   ├── voting_logic.py
│   ├── dashboard_central.py
│   └── dashboard_sync.py
├── templates/
│   └── Test_web.html
├── scripts/
│   ├── install_customer.sh
│   ├── run_machine.sh
│   ├── run_dashboard.sh
│   └── run_sync_worker.sh
└── docs/
    └── LINUX_MACHINE_RUNTIME_SETUP_GUIDE.md
```

หมายเหตุ:

- โฟลเดอร์ `Model/` ต้องอยู่ระดับเดียวกับ `index.py`
- โฟลเดอร์ `modules/` และ `templates/` ต้องไม่เปลี่ยนชื่อ
- ไฟล์ `yolov26s_fixed.pt` ต้องอยู่ที่ root ของโปรเจค
- ถ้าไฟล์หรือโฟลเดอร์สำคัญวางผิดตำแหน่ง โปรแกรมจะรันไม่ขึ้นหรือหา model/config ไม่เจอ

ภาพประกอบที่ควรบันทึก:

`![ภาพขั้นตอนที่ 1 - โครงสร้างโฟลเดอร์หลัก](images/step01-folder-layout.png)`

---

## 3. การดึงโค้ดจาก GitHub

สำหรับเครื่องลูกค้า แนะนำให้ติดตั้งโปรแกรมโดย clone จาก GitHub โดยตรง เพื่อให้โครงสร้างไฟล์ถูกต้องตั้งแต่ต้น

Repository:

```text
https://github.com/visionlabbuu-ship-it/AI-Vision-Wave.git
```

### 3.1 Clone ครั้งแรก

เลือก path ที่ต้องการติดตั้ง เช่น `/home/lab`

```bash
cd /home/lab
git clone https://github.com/visionlabbuu-ship-it/AI-Vision-Wave.git
cd AI-Vision-Wave
pwd
```

สิ่งที่ควรเห็น:

- มีโฟลเดอร์ `/home/lab/AI-Vision-Wave`
- คำสั่ง `pwd` แสดง path ของ repo ที่ clone มา

### 3.2 ตรวจสอบ remote และ branch

```bash
git remote -v
git branch
git status
```

สิ่งที่ควรเห็น:

- `origin` ชี้ไปที่ `https://github.com/visionlabbuu-ship-it/AI-Vision-Wave.git`
- branch หลักเป็น `main`
- ถ้าเพิ่ง clone ใหม่ ควรขึ้นว่า working tree clean

### 3.3 อัปเดตโค้ดจาก GitHub

เมื่อมีการอัปเดตเวอร์ชันใหม่ ให้หยุดโปรแกรมก่อน แล้วรัน:

```bash
cd /home/lab/AI-Vision-Wave
git pull origin main
```

หมายเหตุ:

- ห้ามแก้ไฟล์ source โดยตรงบนเครื่องลูกค้าหากไม่จำเป็น เพราะอาจทำให้ `git pull` ชน conflict
- ค่าที่ต่างกันในแต่ละเครื่องให้เก็บใน `.env.customer`
- ไฟล์ `.env.customer` ไม่ถูก commit ขึ้น GitHub และไม่ควรถูกแชร์พร้อม secret จริง

ภาพประกอบที่ควรบันทึก:

`![ภาพขั้นตอนที่ 2 - clone โค้ดจาก GitHub](images/step02-github-clone.png)`

---

## 4. รายการไฟล์สำคัญที่ต้องมี

ตรวจสอบว่าไฟล์สำคัญต่อไปนี้อยู่ครบก่อนเริ่มรันโปรแกรม

### 4.1 ไฟล์หลัก

- `index.py`
- `modules/`
- `templates/Test_web.html`

### 4.2 ไฟล์ model และ config

- `yolov26s_fixed.pt`
- `calibration_data.json`
- `offsets.json`
- `place_targets.json`
- `Model/CatBoost.joblib`
- `Model/scaler.joblib`
- `Model/label_encoder.joblib`

### 4.3 ไฟล์สคริปต์

- `scripts/run_machine.sh`
- `scripts/run_dashboard.sh`
- `scripts/run_sync_worker.sh`

ใช้คำสั่งนี้เพื่อตรวจสอบเบื้องต้น:

```bash
cd /home/lab/AI-Vision-Wave
ls
ls modules
ls Model
ls scripts
```

สิ่งที่ควรเห็น:

- มีไฟล์ `index.py` ในโฟลเดอร์หลัก
- มีโฟลเดอร์ `modules`, `Model`, `templates`, `scripts`
- มีไฟล์ model อยู่ใน `Model/`

ภาพประกอบที่ควรบันทึก:

`![ภาพขั้นตอนที่ 3 - ตรวจสอบไฟล์สำคัญ](images/step03-check-required-files.png)`

---

## 5. ตัวอย่าง path ที่แนะนำสำหรับติดตั้ง

แนะนำให้เก็บโปรแกรมไว้ใน path ที่สั้นและชัดเจน เช่น

```bash
/home/lab/AI-Vision-Wave
```

หรือ

```bash
/opt/AI-Vision-Wave
```

ไม่แนะนำ:

- path ที่มีเว้นวรรคจำนวนมาก
- path ที่ซ้อนหลายชั้นเกินไป
- path ที่ผู้ใช้ไม่มีสิทธิ์เขียน

ตัวอย่างการย้ายโฟลเดอร์:

```bash
mv AI-Vision-Wave /home/lab/AI-Vision-Wave
cd /home/lab/AI-Vision-Wave
pwd
```

สิ่งที่ควรเห็น:

- คำสั่ง `pwd` แสดง path ที่ต้องการใช้งานจริง

ภาพประกอบที่ควรบันทึก:

`![ภาพขั้นตอนที่ 4 - ตำแหน่งติดตั้งโปรแกรม](images/step04-install-path.png)`

---

## 6. ขั้นตอนสร้าง shell script สำหรับเรียกไฟล์หลัก

ถ้าต้องการอธิบายวิธีสร้างสคริปต์ด้วยตนเอง สามารถใช้ตัวอย่างด้านล่าง

### 6.1 สร้างไฟล์สคริปต์

```bash
cd /home/lab/AI-Vision-Wave
mkdir -p scripts
nano scripts/run_machine.sh
```

### 6.2 วางโค้ดต่อไปนี้ลงในไฟล์

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$ROOT_DIR/.env.customer" ]]; then
  source "$ROOT_DIR/.env.customer"
fi

if [[ -n "${MACHINE_PYTHON:-}" ]]; then
  PYTHON_CMD="$MACHINE_PYTHON"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_CMD="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/Orin_venv/bin/python" ]]; then
  PYTHON_CMD="$ROOT_DIR/Orin_venv/bin/python"
else
  PYTHON_CMD="python3"
fi

cd "$ROOT_DIR"
exec "$PYTHON_CMD" index.py
```

คำอธิบายสั้น ๆ:

- `ROOT_DIR` ใช้หา root ของโปรเจคอัตโนมัติ
- ถ้ามี `.env.customer` จะโหลดค่าตัวแปรก่อน
- ถ้ามี `.venv` หรือ `Orin_venv` จะเลือก Python จาก environment นั้น
- บรรทัดสุดท้ายคือการเรียก `index.py`

ภาพประกอบที่ควรบันทึก:

`![ภาพขั้นตอนที่ 5 - สร้างไฟล์ run_machine.sh](images/step05-create-run-machine-script.png)`

---

## 7. ตั้งสิทธิ์ให้ shell script รันได้

หลังจากบันทึกไฟล์แล้ว ให้ตั้งสิทธิ์ executable

```bash
cd /home/lab/AI-Vision-Wave
chmod +x scripts/run_machine.sh
ls -l scripts/run_machine.sh
```

สิ่งที่ควรเห็น:

- ไฟล์ `run_machine.sh` มีสิทธิ์ `x` เช่น `-rwxr-xr-x`

ภาพประกอบที่ควรบันทึก:

`![ภาพขั้นตอนที่ 6 - ตั้งสิทธิ์ executable](images/step06-chmod-script.png)`

---

## 8. ทดลองรันโปรแกรมหลัก

ใช้คำสั่ง:

```bash
cd /home/lab/AI-Vision-Wave
./scripts/run_machine.sh
```

สิ่งที่ควรเห็น:

- โปรแกรมเริ่มเปิดหน้าต่างหลักของระบบ
- ถ้าเครื่องเชื่อมต่อ hardware ครบ จะเริ่ม initialize camera, detector, robot, spectrum

ถ้าต้องการตรวจสอบก่อนว่าระบบใช้ Python ตัวไหน:

```bash
which python3
./scripts/run_machine.sh
```

ภาพประกอบที่ควรบันทึก:

`![ภาพขั้นตอนที่ 7 - เรียกโปรแกรมด้วย shell script](images/step07-run-machine-script.png)`

---

## 9. การเตรียมไฟล์ `.env.customer`

ถ้าต้องการกำหนด Python path หรือค่าที่เกี่ยวข้องกับการ sync/dashboard ให้สร้างไฟล์ `.env.customer`

```bash
cd /home/lab/AI-Vision-Wave
cp .env.customer.example .env.customer
nano .env.customer
```

ตัวอย่างค่าที่สำคัญ:

```bash
MACHINE_PYTHON=/home/lab/AI-Vision-Wave/Orin_venv/bin/python
DASHBOARD_URL=http://192.168.1.10:5000
DASHBOARD_API_KEY=strong-secret
MACHINE_ID=SORTER-01
MACHINE_NAME=Sorter 01
SITE_NAME=Factory A
LINE_NAME=Line 1
MACHINE_DB_PATH=system_data.db
SYNC_DB_PATH=machine_sync.db
SYNC_INTERVAL_SECONDS=60
```

หมายเหตุ:

- ถ้าต้องการรันเฉพาะ `index.py` อย่างเดียว ค่า sync/dashboard ยังไม่จำเป็นต้องกรอกครบ
- ถ้าต้องการใช้ script เดิมที่สร้างไว้โดยไม่แก้โค้ด ให้กำหนด `MACHINE_PYTHON` ในไฟล์นี้

ภาพประกอบที่ควรบันทึก:

`![ภาพขั้นตอนที่ 8 - แก้ไขไฟล์ .env.customer](images/step08-edit-env-customer.png)`

---

## 10. Checklist ก่อนเริ่มใช้งานจริง

ให้ตรวจสอบรายการต่อไปนี้ทุกครั้งก่อนรัน

- โฟลเดอร์โปรเจคอยู่ใน path ที่ถูกต้อง
- มีไฟล์ `index.py` อยู่ใน root ของโปรเจค
- มีไฟล์ `yolov26s_fixed.pt`
- มีโฟลเดอร์ `Model/` พร้อมไฟล์ `.joblib`
- มีไฟล์ `calibration_data.json`
- มีไฟล์ `offsets.json`
- มี shell script `scripts/run_machine.sh`
- ตั้งสิทธิ์ `chmod +x` แล้ว
- Python environment ที่ใช้รันถูกต้อง
- อุปกรณ์ hardware เชื่อมต่อพร้อมใช้งาน

ภาพประกอบที่ควรบันทึก:

`![ภาพขั้นตอนที่ 9 - Checklist ก่อนรัน](images/step09-preflight-checklist.png)`

---

## 11. ปัญหาที่พบบ่อยและแนวทางตรวจสอบ

### 11.1 รันแล้วขึ้นว่าไม่พบ `index.py`

สาเหตุ:

- รันสคริปต์จาก path ผิด
- โครงสร้างโฟลเดอร์ไม่ตรง

ตรวจสอบ:

```bash
cd /home/lab/AI-Vision-Wave
ls index.py
pwd
```

### 11.2 รันแล้วไม่พบ model

สาเหตุ:

- ไม่มีไฟล์ `yolov26s_fixed.pt`
- วาง model ไม่ถูก path

ตรวจสอบ:

```bash
ls /home/lab/AI-Vision-Wave/yolov26s_fixed.pt
```

### 11.3 รันแล้วไม่พบไฟล์ใน `Model/`

สาเหตุ:

- ไม่มีไฟล์ `.joblib`
- โฟลเดอร์ `Model/` อยู่ผิดตำแหน่ง

ตรวจสอบ:

```bash
ls /home/lab/AI-Vision-Wave/Model
```

### 11.4 รัน script ไม่ได้

สาเหตุ:

- ยังไม่ได้ `chmod +x`

ตรวจสอบ:

```bash
ls -l scripts/run_machine.sh
chmod +x scripts/run_machine.sh
```

### 11.5 ใช้ Python ผิดตัว

ตรวจสอบ:

```bash
which python3
cat .env.customer
```

ภาพประกอบที่ควรบันทึก:

`![ภาพขั้นตอนที่ 10 - แก้ปัญหาที่พบบ่อย](images/step10-troubleshooting.png)`

---

## 12. รายการภาพที่ควรเตรียมเพื่อแปลงเป็น Word

แนะนำให้สร้างโฟลเดอร์สำหรับเก็บภาพประกอบ เช่น

```bash
mkdir -p docs/images
```

รายการภาพที่ควรเตรียม:

1. ภาพโครงสร้างโฟลเดอร์หลัก
2. ภาพ clone โค้ดจาก GitHub
3. ภาพตรวจสอบไฟล์สำคัญ
4. ภาพ path ติดตั้งโปรแกรม
5. ภาพหน้าจอขณะสร้าง `run_machine.sh`
6. ภาพหน้าจอหลัง `chmod +x`
7. ภาพหน้าจอขณะรัน `./scripts/run_machine.sh`
8. ภาพหน้าจอการแก้ไข `.env.customer`
9. ภาพ checklist ก่อนรัน
10. ภาพตัวอย่าง troubleshooting

---

## 13. สรุป

ถ้าผู้ใช้งาน clone โค้ดจาก GitHub ถูกต้อง จัดวางโครงสร้างโฟลเดอร์ถูกต้อง สร้าง `shell script` ถูกต้อง และตั้ง path ของ Python กับ model ครบ ระบบจะสามารถเรียก `index.py` ผ่าน `run_machine.sh` ได้โดยสะดวก และสามารถนำภาพหน้าจอในแต่ละขั้นตอนจากคู่มือนี้ไปจัดทำเป็นเอกสาร Word ได้ต่อทันที
