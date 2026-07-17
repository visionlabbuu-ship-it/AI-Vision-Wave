# AI Vision Wave - Automated Waste Sorting System

ระบบ `AI Vision Wave` เป็นระบบคัดแยกขยะอัตโนมัติสำหรับเครื่อง Linux/Jetson ที่ใช้กล้อง RGB-D, YOLO segmentation, depth measurement, spectrum sensor และแขนกล delta robot เพื่อระบุชนิดวัตถุ ติดตามตำแหน่งบนสายพาน หยิบวัตถุ และคัดแยกลงตำแหน่งที่กำหนด

โปรเจคนี้ประกอบด้วย 2 ส่วนหลัก:

1. โปรแกรมควบคุมเครื่องคัดแยกหลักผ่าน `index.py`
2. ระบบ central dashboard สำหรับรับข้อมูลสรุปราย 1 นาทีจากหลายเครื่องผ่าน `dashboard_app.py` และ `run_dashboard_sync.py`

---

## Table Of Contents

- [System Overview](#system-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Important Runtime Files](#important-runtime-files)
- [Quick Start For Linux/Jetson](#quick-start-for-linuxjetson)
- [Central Dashboard Setup](#central-dashboard-setup)
- [Machine Sync Worker](#machine-sync-worker)
- [Configuration](#configuration)
- [Customer Shell Scripts](#customer-shell-scripts)
- [Testing And Verification](#testing-and-verification)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [GitHub Delivery Notes](#github-delivery-notes)

---

## System Overview

ระบบทำงานเป็น pipeline ต่อเนื่องตั้งแต่การรับภาพจากกล้องจนถึงการควบคุมแขนกล:

```text
RealSense Camera
  -> YOLO Segmentation
  -> Depth / Height Measurement
  -> Object Tracking
  -> Belt Speed Estimation
  -> Picking Queue
  -> Predictive Robot Pick
  -> Spectrum Scan
  -> Bayesian Fusion
  -> Sort To Target Bin
  -> Local Log + Central Dashboard Sync
```

ชนิดวัตถุหลักที่ระบบรองรับ:

| Class | Meaning |
| --- | --- |
| `Glass` | แก้ว |
| `Metal` | โลหะ |
| `Paper` | กระดาษ |
| `Plastic` | พลาสติก |

---

## Key Features

- RealSense RGB-D camera pipeline สำหรับภาพสีและ depth
- YOLO segmentation สำหรับตรวจจับและแยก mask ของวัตถุ
- Depth-based height measurement เพื่อช่วยคำนวณจุดหยิบ
- Centroid tracking สำหรับติดตามวัตถุบนสายพาน
- Vision-based belt speed estimation
- Smart picking queue และ predictive pick สำหรับแขนกล
- Spectrum sensor classification ด้วยโมเดล CatBoost
- Bayesian fusion สำหรับรวมผลจาก vision และ spectrum
- Local SQLite logging สำหรับข้อมูลการคัดแยก
- Central dashboard database แยกจากฐานข้อมูลเครื่องหลัก
- Sync worker ส่งข้อมูลสรุปทุก 1 นาทีจากแต่ละเครื่องไปยัง dashboard กลาง
- Shell scripts สำหรับติดตั้งและรันบน Linux/Jetson

---

## System Architecture

ระบบถูกแบ่งเป็น 3 runtime role เพื่อให้เครื่องหลักยังทำงานได้แม้ dashboard หรือ network มีปัญหา:

| Runtime | File | Purpose |
| --- | --- | --- |
| Machine Runtime | `index.py` | โปรแกรมหลักสำหรับควบคุมกล้อง, detector, tracker, robot, spectrum และ UI |
| Sync Worker | `run_dashboard_sync.py` | อ่าน local database แล้วส่ง summary ราย 1 นาทีไป dashboard |
| Central Dashboard | `dashboard_app.py` | Flask web dashboard พร้อม central SQLite database |

การแยกส่วนนี้ช่วยให้หลายเครื่องคัดแยกสามารถส่งข้อมูลเข้าหน้าเว็บกลางเดียวกันได้ โดยไม่ให้เว็บกลางต้อง query ฐานข้อมูลในแต่ละเครื่องโดยตรง

---

## Repository Structure

โครงสร้างไฟล์หลักที่ใช้จริง:

```text
AI-Vision-Wave/
├── index.py
├── dashboard_app.py
├── run_dashboard_sync.py
├── yolov26s_fixed.pt
├── calibration_data.json
├── offsets.json
├── place_targets.json
├── req_final.txt
├── requirements-dashboard.txt
├── .env.customer.example
├── Model/
│   ├── CatBoost.joblib
│   ├── scaler.joblib
│   ├── label_encoder.joblib
│   └── used_features.joblib
├── modules/
│   ├── camera.py
│   ├── config.py
│   ├── dashboard_central.py
│   ├── dashboard_sync.py
│   ├── database.py
│   ├── detector.py
│   ├── robot.py
│   ├── spectrum.py
│   ├── tracker.py
│   └── voting_logic.py
├── scripts/
│   ├── install_customer.sh
│   ├── run_machine.sh
│   ├── run_dashboard.sh
│   └── run_sync_worker.sh
├── templates/
│   └── Test_web.html
├── tests/
│   ├── test_dashboard_app.py
│   └── test_dashboard_sync.py
└── docs/
    ├── CUSTOMER_DEPLOYMENT_GUIDE.md
    ├── LINUX_MACHINE_RUNTIME_SETUP_GUIDE.md
    ├── PROJECT_SUMMARY.md
    └── TECHNICAL_DOCUMENTATION.md
```

---

## Hardware Requirements

ระบบถูกออกแบบสำหรับเครื่อง Linux/Jetson ที่มีอุปกรณ์หลักดังนี้:

| Component | Usage |
| --- | --- |
| NVIDIA Jetson | เครื่องประมวลผลหลัก |
| Intel RealSense RGB-D Camera | ตรวจจับวัตถุและวัด depth |
| Delta Robot | หยิบและคัดแยกวัตถุ |
| Linear Slider | เคลื่อนตำแหน่งแขนกลตามแนวระบบ |
| AS7265X Spectrum Sensor | ตรวจสอบชนิดวัสดุด้วย spectrum |
| Conveyor Belt | ลำเลียงวัตถุผ่านพื้นที่ตรวจจับ |
| Vacuum Gripper | จับวัตถุระหว่าง pick cycle |

พอร์ตอุปกรณ์จริงอาจต้องปรับใน `modules/config.py` ให้ตรงกับเครื่องหน้างาน

---

## Software Requirements

แพลตฟอร์มเป้าหมาย:

- Linux / Jetson
- Python 3.10
- RealSense SDK / `pyrealsense2`
- OpenCV
- PyTorch / Ultralytics
- Dear PyGui
- SQLite
- Flask สำหรับ dashboard

ไฟล์ dependency หลัก:

| File | Purpose |
| --- | --- |
| `req_final.txt` | dependency ฝั่ง machine runtime |
| `requirements-dashboard.txt` | dependency ขั้นต่ำสำหรับ dashboard และ sync worker |

หมายเหตุ: runtime ของเครื่องหลักอาจต้องใช้ wheel หรือ dependency เฉพาะ Jetson ที่ต้อง provision แยกตามเครื่องจริง

---

## Important Runtime Files

ไฟล์เหล่านี้ต้องอยู่ในตำแหน่งถูกต้องก่อนรันระบบ:

| File / Directory | Required For |
| --- | --- |
| `index.py` | machine runtime |
| `modules/` | core application modules |
| `yolov26s_fixed.pt` | YOLO segmentation model |
| `Model/CatBoost.joblib` | spectrum classification |
| `Model/scaler.joblib` | spectrum preprocessing |
| `Model/label_encoder.joblib` | spectrum label mapping |
| `calibration_data.json` | camera/ROI calibration |
| `offsets.json` | robot picking offsets |
| `place_targets.json` | target bin positions |
| `templates/Test_web.html` | dashboard web UI |

---

## Quick Start For Linux/Jetson

ขั้นตอนนี้ใช้สำหรับเครื่องคัดแยกหลัก

1. Clone repository

```bash
git clone https://github.com/visionlabbuu-ship-it/AI-Vision-Wave.git
cd AI-Vision-Wave
```

2. ตรวจสอบไฟล์สำคัญ

```bash
ls index.py
ls modules
ls Model
ls yolov26s_fixed.pt
```

3. ตั้งสิทธิ์ shell scripts

```bash
chmod +x scripts/*.sh
```

4. สร้าง config สำหรับลูกค้า

```bash
cp .env.customer.example .env.customer
nano .env.customer
```

5. รันโปรแกรมหลัก

```bash
./scripts/run_machine.sh
```

คู่มือแบบละเอียดสำหรับผู้ใช้งาน Linux/Jetson อยู่ที่ [docs/LINUX_MACHINE_RUNTIME_SETUP_GUIDE.md](docs/LINUX_MACHINE_RUNTIME_SETUP_GUIDE.md)

---

## Central Dashboard Setup

ใช้กับเครื่อง server หรือเครื่องกลางที่ต้องแสดงผล dashboard

1. ติดตั้ง dependency สำหรับ dashboard

```bash
chmod +x scripts/*.sh
./scripts/install_customer.sh
```

2. แก้ไข `.env.customer`

```bash
DASHBOARD_API_KEY=change-me
CENTRAL_DASHBOARD_DB=central_dashboard.db
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=5000
```

3. เปิด dashboard

```bash
./scripts/run_dashboard.sh
```

4. เปิด browser

```text
http://<server-ip>:5000
```

---

## Machine Sync Worker

sync worker ทำหน้าที่ส่งข้อมูลสรุปราย 1 นาทีจากเครื่องคัดแยกไปยัง central dashboard

ตั้งค่าใน `.env.customer` ฝั่งเครื่องคัดแยก:

```bash
DASHBOARD_URL=http://dashboard-server:5000
DASHBOARD_API_KEY=change-me
MACHINE_ID=SORTER-01
MACHINE_NAME=Sorter 01
SITE_NAME=Factory A
LINE_NAME=Line 1
MACHINE_DB_PATH=system_data.db
SYNC_DB_PATH=machine_sync.db
SYNC_INTERVAL_SECONDS=60
```

รัน worker:

```bash
./scripts/run_sync_worker.sh
```

ทดสอบ sync หนึ่งรอบ:

```bash
./scripts/run_sync_worker.sh --once
```

ถ้า dashboard หรือ network ใช้งานไม่ได้ worker จะเก็บ payload ไว้ใน `machine_sync.db` แล้ว retry รอบถัดไป

---

## Configuration

ไฟล์ config สำคัญ:

| File | Description |
| --- | --- |
| `.env.customer` | runtime setting สำหรับเครื่องลูกค้า ไม่ควร commit |
| `.env.customer.example` | template config ที่ commit ได้ |
| `modules/config.py` | hardware, model path, ROI, robot, belt และ class constants |
| `calibration_data.json` | calibration ของกล้องและ ROI |
| `offsets.json` | offset สำหรับ pick position |
| `place_targets.json` | ตำแหน่งปล่อยวัตถุตาม class |

ตัวอย่าง `.env.customer`:

```bash
MACHINE_PYTHON=/home/lab/AI-Vision-Wave/Orin_venv/bin/python
DASHBOARD_URL=http://192.168.1.10:5000
DASHBOARD_API_KEY=change-me
MACHINE_ID=SORTER-01
MACHINE_NAME=Sorter 01
SITE_NAME=Factory A
LINE_NAME=Line 1
MACHINE_DB_PATH=system_data.db
SYNC_DB_PATH=machine_sync.db
SYNC_INTERVAL_SECONDS=60
```

---

## Customer Shell Scripts

| Script | Description |
| --- | --- |
| `scripts/install_customer.sh` | สร้าง virtualenv สำหรับ dashboard/sync และติดตั้ง dependency |
| `scripts/run_machine.sh` | เรียก `index.py` โดยเลือก Python จาก `.env.customer`, `.venv`, `Orin_venv` หรือ `python3` |
| `scripts/run_dashboard.sh` | เปิด Flask dashboard |
| `scripts/run_sync_worker.sh` | เปิด sync worker หรือรันแบบ `--once` |

บน Linux desktop สามารถตั้งค่าให้ผู้ใช้งาน double-click shell script ได้หลังจาก `chmod +x scripts/*.sh`

---

## Testing And Verification

รัน test สำหรับ dashboard และ sync:

```bash
python -m pytest tests/test_dashboard_app.py tests/test_dashboard_sync.py -q
```

ตรวจ syntax ของ Python:

```bash
python -m py_compile dashboard_app.py run_dashboard_sync.py modules/*.py
```

ตรวจ syntax ของ shell scripts:

```bash
bash -n scripts/install_customer.sh scripts/run_machine.sh scripts/run_dashboard.sh scripts/run_sync_worker.sh
```

---

## Documentation

เอกสารหลักใน `docs/`:

| Document | Purpose |
| --- | --- |
| [docs/CUSTOMER_SIMPLE_USER_GUIDE_TH.md](docs/CUSTOMER_SIMPLE_USER_GUIDE_TH.md) | คู่มือใช้งานแบบย่อสำหรับลูกค้า เปิดโปรแกรม ใช้ UI และหยุดระบบ |
| [docs/LINUX_MACHINE_RUNTIME_SETUP_GUIDE.md](docs/LINUX_MACHINE_RUNTIME_SETUP_GUIDE.md) | คู่มือจัดโครงสร้างไฟล์และรันโปรแกรมบน Linux/Jetson |
| [docs/CUSTOMER_DEPLOYMENT_GUIDE.md](docs/CUSTOMER_DEPLOYMENT_GUIDE.md) | คู่มือ deployment สำหรับลูกค้า |
| [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md) | สรุปภาพรวมระบบและ component |
| [docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md) | เอกสาร technical รายละเอียดเชิงระบบ |
| [docs/superpowers/specs/2026-07-13-central-dashboard-sync-design.md](docs/superpowers/specs/2026-07-13-central-dashboard-sync-design.md) | design spec ของ central dashboard sync |
| [docs/superpowers/plans/2026-07-13-central-dashboard-sync-packaging.md](docs/superpowers/plans/2026-07-13-central-dashboard-sync-packaging.md) | implementation plan ของชุด delivery |

---

## Troubleshooting

### Script รันไม่ได้

ตรวจสิทธิ์:

```bash
ls -l scripts/run_machine.sh
chmod +x scripts/run_machine.sh
```

### ไม่พบ `index.py`

ตรวจ path:

```bash
pwd
ls index.py
```

ต้องรันจาก root ของโปรเจค หรือใช้ `scripts/run_machine.sh`

### ไม่พบ YOLO model

ตรวจไฟล์:

```bash
ls yolov26s_fixed.pt
```

### ไม่พบ spectrum model

ตรวจโฟลเดอร์ `Model/`:

```bash
ls Model
```

### Dashboard ไม่มีข้อมูล

ตรวจว่า sync worker ทำงานอยู่:

```bash
./scripts/run_sync_worker.sh --once
```

ตรวจค่าใน `.env.customer`:

```bash
cat .env.customer
```

ค่าที่ต้องตรงกันคือ `DASHBOARD_URL` และ `DASHBOARD_API_KEY`

### Push GitHub แล้วไฟล์ใหญ่เกิน

repo นี้ตั้ง `.gitignore` เพื่อกันไฟล์ dataset, database, zip, log และ local runtime artifacts แล้ว ถ้ามีไฟล์ใหญ่หลุดเข้า history ต้องลบออกจาก Git history ก่อน push

---

## GitHub Delivery Notes

ไฟล์ที่ไม่ควร commit:

- `.env.customer`
- `*.db`
- `*.zip`
- dataset folders
- runtime logs
- recordings
- telemetry CSV
- local virtual environments
- `.git.backup/`

ไฟล์ model ขนาดเล็กที่จำเป็นต่อ runtime ถูก commit ไว้แล้ว:

- `yolov26s_fixed.pt`
- `Model/CatBoost.joblib`
- `Model/scaler.joblib`
- `Model/label_encoder.joblib`
- `Model/used_features.joblib`

---

## License And Ownership

โปรเจคนี้จัดทำสำหรับระบบคัดแยกขยะอัตโนมัติของทีมพัฒนา AI Vision Wave / Vision Lab BUU หากนำไปใช้งานต่อ ควรตรวจสอบสิทธิ์การใช้งานของ hardware driver, model weights, dependency และ dataset ที่เกี่ยวข้องก่อนเผยแพร่ภายนอกองค์กร
