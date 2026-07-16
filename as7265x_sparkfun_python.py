# as7265x_sparkfun_python.py
# Python port of SparkFun AS7265X Arduino Library
#
# - Logic register-level ตาม Arduino C++ (SparkFun_AS7265X.h / .cpp)
# - ใช้ I2C ผ่าน smbus2
# - รองรับ 18 channels (A–F, G–L, R–W) ทั้ง raw + calibrated
#
# การใช้งานบน Jetson Orin Nano:
#
#   from as7265x_sparkfun_python import AS7265X, AS7265X_GAIN_64X, AS7265X_LED_CURRENT_LIMIT_12_5MA
#   from as7265x_sparkfun_python import AS7265x_LED_IR
#
#   sensor = AS7265X(i2c_bus=1)  # ถ้าเซนเซอร์อยู่บน /dev/i2c-1
#
#   if not sensor.begin():
#       print("ERROR: AS7265X not detected")
#   else:
#       sensor.setGain(AS7265X_GAIN_64X)
#       sensor.setIntegrationCycles(25)
#       sensor.setBulbCurrent(AS7265X_LED_CURRENT_LIMIT_12_5MA, AS7265x_LED_IR)
#       sensor.enableBulb(AS7265x_LED_IR)
#       sensor.takeMeasurements()
#       sensor.disableBulb(AS7265x_LED_IR)
#       print("A:", sensor.getCalibratedA())
#       print("G:", sensor.getCalibratedG())
#       print("R:", sensor.getCalibratedR())
#

import time
import struct

from smbus2 import SMBus



# ---------------------------
# I2C base address
# ---------------------------
AS7265X_ADDR = 0x49  # SparkFun AS7265x default I2C address

# ---------------------------
# Virtual register interface
# ---------------------------
AS7265X_STATUS_REG = 0x00
AS7265X_WRITE_REG = 0x01
AS7265X_READ_REG = 0x02

AS7265X_TX_VALID = 0x02  # bit1
AS7265X_RX_VALID = 0x01  # bit0

# ---------------------------
# Register map
# ---------------------------
AS7265X_HW_VERSION_HIGH = 0x00
AS7265X_HW_VERSION_LOW = 0x01
AS7265X_FW_VERSION_HIGH = 0x02
AS7265X_FW_VERSION_LOW = 0x03

AS7265X_CONFIG = 0x04
AS7265X_INTERGRATION_TIME = 0x05  # (สะกดเหมือนต้นฉบับ Arduino)
AS7265X_DEVICE_TEMP = 0x06
AS7265X_LED_CONFIG = 0x07

# Raw data registers (each is high,low)
AS7265X_R_G_A = 0x08
AS7265X_S_H_B = 0x0A
AS7265X_T_I_C = 0x0C
AS7265X_U_J_D = 0x0E
AS7265X_V_K_E = 0x10
AS7265X_W_L_F = 0x12

# Calibrated float (4 bytes each)
AS7265X_R_G_A_CAL = 0x14
AS7265X_S_H_B_CAL = 0x18
AS7265X_T_I_C_CAL = 0x1C
AS7265X_U_J_D_CAL = 0x20
AS7265X_V_K_E_CAL = 0x24
AS7265X_W_L_F_CAL = 0x28

AS7265X_DEV_SELECT_CONTROL = 0x4F

# Coef registers (เผื่อใช้งานภายหลัง)
AS7265X_COEF_DATA_0 = 0x50
AS7265X_COEF_DATA_1 = 0x51
AS7265X_COEF_DATA_2 = 0x52
AS7265X_COEF_DATA_3 = 0x53
AS7265X_COEF_DATA_READ = 0x54
AS7265X_COEF_DATA_WRITE = 0x55

# Polling delay between status checks (ms)
# Arduino uses 5ms but Jetson I2C is much faster; 0.5ms is safe
AS7265X_POLLING_DELAY = 0.5

# ---------------------------
# Device select values
# ---------------------------
AS72651_NIR = 0x00      # x51 (NIR)
AS72652_VISIBLE = 0x01  # x52 (VIS)
AS72653_UV = 0x02       # x53 (UV)

# ---------------------------
# LED / Bulb selection
# ---------------------------
AS7265x_LED_WHITE = 0x00  # onboard white LED (x51)
AS7265x_LED_IR = 0x01     # onboard IR LED (x52)
AS7265x_LED_UV = 0x02     # onboard UV LED (x53)

# Bulb current limits
AS7265X_LED_CURRENT_LIMIT_12_5MA = 0b00
AS7265X_LED_CURRENT_LIMIT_25MA = 0b01
AS7265X_LED_CURRENT_LIMIT_50MA = 0b10
AS7265X_LED_CURRENT_LIMIT_100MA = 0b11

# Indicator LED current limits
AS7265X_INDICATOR_CURRENT_LIMIT_1MA = 0b00
AS7265X_INDICATOR_CURRENT_LIMIT_2MA = 0b01
AS7265X_INDICATOR_CURRENT_LIMIT_4MA = 0b10
AS7265X_INDICATOR_CURRENT_LIMIT_8MA = 0b11

# Gain
AS7265X_GAIN_1X = 0b00
AS7265X_GAIN_37X = 0b01
AS7265X_GAIN_16X = 0b10
AS7265X_GAIN_64X = 0b11

# Measurement modes
AS7265X_MEASUREMENT_MODE_4CHAN = 0b00
AS7265X_MEASUREMENT_MODE_4CHAN_2 = 0b01
AS7265X_MEASUREMENT_MODE_6CHAN_CONTINUOUS = 0b10
AS7265X_MEASUREMENT_MODE_6CHAN_ONE_SHOT = 0b11


class AS7265X:
    """
    Python version of SparkFun AS7265X class (Arduino library).

    - ทำงานระดับ virtual register เหมือน Arduino
    - ใช้กับบอร์ด SparkFun AS7265x Triad ได้ตรง ๆ
    """

    def __init__(self, i2c_bus=1, address=AS7265X_ADDR, bus=None):
        """
        i2c_bus: หมายเลข I2C bus (เช่น 1 สำหรับ /dev/i2c-1 บน Jetson)
        address: I2C address (default 0x49)
        bus:     ถ้ามี SMBus อยู่แล้ว สามารถส่งมาร่วมใช้ได้
        """
        self.address = address
        self.bus = bus if bus is not None else SMBus(i2c_bus)

        # maxWaitTime ใน Arduino ถูกคำนวณจาก integration time
        # ให้ค่าเริ่มต้นตามตัวอย่าง 2.8 * 255 * 1.5 ≈ 1071ms
        self.maxWaitTime = 1071

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _millis(self):
        return int(time.monotonic() * 1000)

    def _delay_ms(self, ms):
        time.sleep(ms / 1000.0)

    # ------------------------------------------------------------------
    # Public API (พอร์ตจาก Arduino)
    # ------------------------------------------------------------------
    def begin(self):
        """
        Initialize sensor (เหมือน sensor.begin() ใน Arduino)
        Return True ถ้าเจออุปกรณ์และ slaves พร้อมใช้งาน
        """
        if not self.isConnected():
            return False

        # เช็คว่าทั้ง x52 และ x53 ถูก detect หรือไม่ (เหมือน Arduino)
        value = self.virtualReadRegister(AS7265X_DEV_SELECT_CONTROL)
        # บิต 4 และ 5 คือ detect flags
        if (value & 0b00110000) == 0:
            return False

        # Default configuration เหมือน Arduino library
        self.setBulbCurrent(AS7265X_LED_CURRENT_LIMIT_12_5MA,
                            AS7265x_LED_WHITE)
        self.setBulbCurrent(AS7265X_LED_CURRENT_LIMIT_12_5MA,
                            AS7265x_LED_IR)
        self.setBulbCurrent(AS7265X_LED_CURRENT_LIMIT_12_5MA,
                            AS7265x_LED_UV)

        self.disableBulb(AS7265x_LED_WHITE)
        self.disableBulb(AS7265x_LED_IR)
        self.disableBulb(AS7265x_LED_UV)

        self.setIndicatorCurrent(AS7265X_INDICATOR_CURRENT_LIMIT_8MA)
        self.enableIndicator()

        # Integration time:
        #   T_int(ms) = 2.8 * (value + 1)  (อิง SparkFun)
        # ที่นี่ใช้ 49 เหมือน Arduino → ~140ms
        self.setIntegrationCycles(49)

        self.setGain(AS7265X_GAIN_64X)
        self.setMeasurementMode(AS7265X_MEASUREMENT_MODE_6CHAN_ONE_SHOT)
        self.enableInterrupt()

        return True

    def isConnected(self):
        """
        เช็คว่า I2C address ตอบสนองหรือไม่
        """
        for _ in range(10):
            try:
                self.bus.read_byte_data(self.address, AS7265X_STATUS_REG)
                return True
            except OSError:
                self._delay_ms(10)
        return False

    # ---------------- Firmware / Hardware info ----------------
    def getDeviceType(self):
        return self.virtualReadRegister(AS7265X_HW_VERSION_HIGH)

    def getHardwareVersion(self):
        return self.virtualReadRegister(AS7265X_HW_VERSION_LOW)

    def getMajorFirmwareVersion(self):
        self.virtualWriteRegister(AS7265X_FW_VERSION_HIGH, 0x01)
        self.virtualWriteRegister(AS7265X_FW_VERSION_LOW, 0x01)
        return self.virtualReadRegister(AS7265X_FW_VERSION_LOW)

    def getPatchFirmwareVersion(self):
        self.virtualWriteRegister(AS7265X_FW_VERSION_HIGH, 0x02)
        self.virtualWriteRegister(AS7265X_FW_VERSION_LOW, 0x02)
        return self.virtualReadRegister(AS7265X_FW_VERSION_LOW)

    def getBuildFirmwareVersion(self):
        self.virtualWriteRegister(AS7265X_FW_VERSION_HIGH, 0x03)
        self.virtualWriteRegister(AS7265X_FW_VERSION_LOW, 0x03)
        return self.virtualReadRegister(AS7265X_FW_VERSION_LOW)

    # ---------------- Measurement control ----------------
    def takeMeasurements(self):
        """
        สั่งวัดแบบ one-shot 6 channel (ตาม mode)
        รอจน dataAvailable หรือ timeout
        """
        self.setMeasurementMode(AS7265X_MEASUREMENT_MODE_6CHAN_ONE_SHOT)

        start_time = self._millis()
        while not self.dataAvailable():
            if self._millis() - start_time > self.maxWaitTime:
                return  # timeout
            self._delay_ms(AS7265X_POLLING_DELAY)

    def takeMeasurementsWithBulb(self):
        """
        เปิดหลอดทั้ง 3, วัด, ปิดหลอด (เหมือน Arduino)
        """
        self.enableBulb(AS7265x_LED_WHITE)
        self.enableBulb(AS7265x_LED_IR)
        self.enableBulb(AS7265x_LED_UV)

        self.takeMeasurements()

        self.disableBulb(AS7265x_LED_WHITE)
        self.disableBulb(AS7265x_LED_IR)
        self.disableBulb(AS7265x_LED_UV)

    # ---------------- Batch calibrated read (fast) --------
    def getAllCalibratedValues(self):
        """
        Read all 18 calibrated channels in one call, grouped by device
        to minimise selectDevice switches.
        Returns list of 18 floats in the same order as getCalibratedA..L.
        Order: A,B,C,D,E,F (UV) / G,H,I,J,K,L (VIS) / R,S,T,U,V,W (NIR)
        """
        cal_regs = [
            AS7265X_R_G_A_CAL, AS7265X_S_H_B_CAL, AS7265X_T_I_C_CAL,
            AS7265X_U_J_D_CAL, AS7265X_V_K_E_CAL, AS7265X_W_L_F_CAL,
        ]
        vals = []
        # Read each device group in sequence (3 devices × 6 channels)
        for device in [AS72653_UV, AS72652_VISIBLE, AS72651_NIR]:
            self.selectDevice(device)
            for reg in cal_regs:
                b = [
                    self.virtualReadRegister(reg + 0),
                    self.virtualReadRegister(reg + 1),
                    self.virtualReadRegister(reg + 2),
                    self.virtualReadRegister(reg + 3),
                ]
                vals.append(struct.unpack('>f', bytes(b))[0])
        return vals  # [A,B,C,D,E,F, G,H,I,J,K,L, R,S,T,U,V,W]

    # ---------------- Raw readings helpers ----------------
    def _getChannel(self, channelRegister, device):
        self.selectDevice(device)
        high = self.virtualReadRegister(channelRegister)
        low = self.virtualReadRegister(channelRegister + 1)
        return (high << 8) | low

    # Raw (UV = x53)
    def getA(self): return self._getChannel(AS7265X_R_G_A, AS72653_UV)
    def getB(self): return self._getChannel(AS7265X_S_H_B, AS72653_UV)
    def getC(self): return self._getChannel(AS7265X_T_I_C, AS72653_UV)
    def getD(self): return self._getChannel(AS7265X_U_J_D, AS72653_UV)
    def getE(self): return self._getChannel(AS7265X_V_K_E, AS72653_UV)
    def getF(self): return self._getChannel(AS7265X_W_L_F, AS72653_UV)

    # Raw (Visible = x52)
    def getG(self): return self._getChannel(AS7265X_R_G_A, AS72652_VISIBLE)
    def getH(self): return self._getChannel(AS7265X_S_H_B, AS72652_VISIBLE)
    def getI(self): return self._getChannel(AS7265X_T_I_C, AS72652_VISIBLE)
    def getJ(self): return self._getChannel(AS7265X_U_J_D, AS72652_VISIBLE)
    def getK(self): return self._getChannel(AS7265X_V_K_E, AS72652_VISIBLE)
    def getL(self): return self._getChannel(AS7265X_W_L_F, AS72652_VISIBLE)

    # Raw (NIR = x51)
    def getR(self): return self._getChannel(AS7265X_R_G_A, AS72651_NIR)
    def getS(self): return self._getChannel(AS7265X_S_H_B, AS72651_NIR)
    def getT(self): return self._getChannel(AS7265X_T_I_C, AS72651_NIR)
    def getU(self): return self._getChannel(AS7265X_U_J_D, AS72651_NIR)
    def getV(self): return self._getChannel(AS7265X_V_K_E, AS72651_NIR)
    def getW(self): return self._getChannel(AS7265X_W_L_F, AS72651_NIR)

    # ---------------- Calibrated readings helpers ----------------
    def _getCalibratedValue(self, calAddress, device):
        """
        อ่าน float (4 bytes, big-endian) จาก address ที่กำหนด
        """
        self.selectDevice(device)
        b = [
            self.virtualReadRegister(calAddress + 0),
            self.virtualReadRegister(calAddress + 1),
            self.virtualReadRegister(calAddress + 2),
            self.virtualReadRegister(calAddress + 3),
        ]
        return struct.unpack('>f', bytes(b))[0]  # big-endian float

    # Calibrated (UV)
    def getCalibratedA(self): return self._getCalibratedValue(AS7265X_R_G_A_CAL, AS72653_UV)
    def getCalibratedB(self): return self._getCalibratedValue(AS7265X_S_H_B_CAL, AS72653_UV)
    def getCalibratedC(self): return self._getCalibratedValue(AS7265X_T_I_C_CAL, AS72653_UV)
    def getCalibratedD(self): return self._getCalibratedValue(AS7265X_U_J_D_CAL, AS72653_UV)
    def getCalibratedE(self): return self._getCalibratedValue(AS7265X_V_K_E_CAL, AS72653_UV)
    def getCalibratedF(self): return self._getCalibratedValue(AS7265X_W_L_F_CAL, AS72653_UV)

    # Calibrated (Visible)
    def getCalibratedG(self): return self._getCalibratedValue(AS7265X_R_G_A_CAL, AS72652_VISIBLE)
    def getCalibratedH(self): return self._getCalibratedValue(AS7265X_S_H_B_CAL, AS72652_VISIBLE)
    def getCalibratedI(self): return self._getCalibratedValue(AS7265X_T_I_C_CAL, AS72652_VISIBLE)
    def getCalibratedJ(self): return self._getCalibratedValue(AS7265X_U_J_D_CAL, AS72652_VISIBLE)
    def getCalibratedK(self): return self._getCalibratedValue(AS7265X_V_K_E_CAL, AS72652_VISIBLE)
    def getCalibratedL(self): return self._getCalibratedValue(AS7265X_W_L_F_CAL, AS72652_VISIBLE)

    # Calibrated (NIR)
    def getCalibratedR(self): return self._getCalibratedValue(AS7265X_R_G_A_CAL, AS72651_NIR)
    def getCalibratedS(self): return self._getCalibratedValue(AS7265X_S_H_B_CAL, AS72651_NIR)
    def getCalibratedT(self): return self._getCalibratedValue(AS7265X_T_I_C_CAL, AS72651_NIR)
    def getCalibratedU(self): return self._getCalibratedValue(AS7265X_U_J_D_CAL, AS72651_NIR)
    def getCalibratedV(self): return self._getCalibratedValue(AS7265X_V_K_E_CAL, AS72651_NIR)
    def getCalibratedW(self): return self._getCalibratedValue(AS7265X_W_L_F_CAL, AS72651_NIR)

    # ---------------- Gain / integration / mode ----------------
    def setMeasurementMode(self, mode):
        if mode > 0b11:
            mode = 0b11
        value = self.virtualReadRegister(AS7265X_CONFIG)
        value &= 0b11110011  # clear measurement bits (2,3)
        value |= (mode << 2)
        self.virtualWriteRegister(AS7265X_CONFIG, value)

    def setGain(self, gain):
        if gain > 0b11:
            gain = 0b11
        value = self.virtualReadRegister(AS7265X_CONFIG)
        value &= 0b11001111  # clear gain bits (4,5)
        value |= (gain << 4)
        self.virtualWriteRegister(AS7265X_CONFIG, value)

    def setIntegrationCycles(self, cycleValue):
        """
        Integration time = 2.8 * (value + 1) ms โดยประมาณ
        """
        self.maxWaitTime = int(cycleValue * 2.8 * 1.5) + 1
        self.virtualWriteRegister(AS7265X_INTERGRATION_TIME, cycleValue & 0xFF)

    # ---------------- Interrupt ----------------
    def enableInterrupt(self):
        value = self.virtualReadRegister(AS7265X_CONFIG)
        value |= (1 << 6)
        self.virtualWriteRegister(AS7265X_CONFIG, value)

    def disableInterrupt(self):
        value = self.virtualReadRegister(AS7265X_CONFIG)
        value &= ~(1 << 6)
        self.virtualWriteRegister(AS7265X_CONFIG, value)

    def dataAvailable(self):
        value = self.virtualReadRegister(AS7265X_CONFIG)
        return (value & (1 << 1)) != 0

    # ---------------- Bulb / Indicator LED ----------------
    def enableBulb(self, device):
        self.selectDevice(device)
        value = self.virtualReadRegister(AS7265X_LED_CONFIG)
        value |= (1 << 3)
        self.virtualWriteRegister(AS7265X_LED_CONFIG, value)

    def disableBulb(self, device):
        self.selectDevice(device)
        value = self.virtualReadRegister(AS7265X_LED_CONFIG)
        value &= ~(1 << 3)
        self.virtualWriteRegister(AS7265X_LED_CONFIG, value)

    def setBulbCurrent(self, current, device):
        self.selectDevice(device)
        if current > 0b11:
            current = 0b11
        value = self.virtualReadRegister(AS7265X_LED_CONFIG)
        value &= 0b11001111  # clear ICL_DRV bits (4,5)
        value |= (current << 4)
        self.virtualWriteRegister(AS7265X_LED_CONFIG, value)

    def enableIndicator(self):
        self.selectDevice(AS72651_NIR)
        value = self.virtualReadRegister(AS7265X_LED_CONFIG)
        value |= (1 << 0)
        self.virtualWriteRegister(AS7265X_LED_CONFIG, value)

    def disableIndicator(self):
        self.selectDevice(AS72651_NIR)
        value = self.virtualReadRegister(AS7265X_LED_CONFIG)
        value &= ~(1 << 0)
        self.virtualWriteRegister(AS7265X_LED_CONFIG, value)

    def setIndicatorCurrent(self, current):
        self.selectDevice(AS72651_NIR)
        if current > 0b11:
            current = 0b11
        value = self.virtualReadRegister(AS7265X_LED_CONFIG)
        value &= 0b11111001  # clear bits 1,2
        value |= (current << 1)
        self.virtualWriteRegister(AS7265X_LED_CONFIG, value)

    # ---------------- Temperature ----------------
    def getTemperature(self, deviceNumber=0):
        """
        0 = x51 (NIR), 1 = x52 (VIS), 2 = x53 (UV)
        """
        self.selectDevice(deviceNumber)
        return self.virtualReadRegister(AS7265X_DEVICE_TEMP)

    def getTemperatureAverage(self):
        total = 0.0
        for dev in range(3):
            total += self.getTemperature(dev)
        return total / 3.0

    # ---------------- Reset ----------------
    def softReset(self):
        value = self.virtualReadRegister(AS7265X_CONFIG)
        value |= (1 << 7)  # RST bit
        self.virtualWriteRegister(AS7265X_CONFIG, value)
        self._delay_ms(1000)

    # ---------------- Device select ----------------
    def selectDevice(self, device):
        """
        เลือก x51/x52/x53 ก่อนอ่าน/เขียน register
        """
        self.virtualWriteRegister(AS7265X_DEV_SELECT_CONTROL, device & 0xFF)

    # ------------------------------------------------------------------
    # Virtual register access (สำคัญมาก)
    # ------------------------------------------------------------------
    def virtualReadRegister(self, virtualAddr):
        """
        อ่าน virtual register ผ่าน STATUS/WRITE/READ เหมือน Arduino
        """
        # เคลียร์ RX ถ้ามีค้าง
        status = self.readRegister(AS7265X_STATUS_REG)
        if (status & AS7265X_RX_VALID) != 0:
            _ = self.readRegister(AS7265X_READ_REG)

        # รอให้ TX ว่าง
        start = self._millis()
        while True:
            if self._millis() - start > self.maxWaitTime:
                return 0
            status = self.readRegister(AS7265X_STATUS_REG)
            if (status & AS7265X_TX_VALID) == 0:
                break
            self._delay_ms(AS7265X_POLLING_DELAY)

        # เขียน virtual address (bit7 = 0 สำหรับ read)
        self.writeRegister(AS7265X_WRITE_REG, virtualAddr & 0x7F)

        # รอให้ RX มี data
        start = self._millis()
        while True:
            if self._millis() - start > self.maxWaitTime:
                return 0
            status = self.readRegister(AS7265X_STATUS_REG)
            if (status & AS7265X_RX_VALID) != 0:
                break
            self._delay_ms(AS7265X_POLLING_DELAY)

        incoming = self.readRegister(AS7265X_READ_REG)
        return incoming

    def virtualWriteRegister(self, virtualAddr, dataToWrite):
        """
        เขียน virtual register ผ่าน STATUS/WRITE
        """
        # รอให้ TX ว่าง
        start = self._millis()
        while True:
            if self._millis() - start > self.maxWaitTime:
                return
            status = self.readRegister(AS7265X_STATUS_REG)
            if (status & AS7265X_TX_VALID) == 0:
                break
            self._delay_ms(AS7265X_POLLING_DELAY)

        # เขียน address (bit7 = 1 สำหรับ write)
        self.writeRegister(AS7265X_WRITE_REG, (virtualAddr | 0x80) & 0xFF)

        # รอให้ TX ว่างอีกรอบ
        start = self._millis()
        while True:
            if self._millis() - start > self.maxWaitTime:
                return
            status = self.readRegister(AS7265X_STATUS_REG)
            if (status & AS7265X_TX_VALID) == 0:
                break
            self._delay_ms(AS7265X_POLLING_DELAY)

        # เขียน data
        self.writeRegister(AS7265X_WRITE_REG, dataToWrite & 0xFF)

    # ------------------------------------------------------------------
    # Low-level I2C
    # ------------------------------------------------------------------
    def readRegister(self, reg):
        try:
            return self.bus.read_byte_data(self.address, reg)
        except OSError:
            return 0

    def writeRegister(self, reg, val):
        try:
            self.bus.write_byte_data(self.address, reg, val & 0xFF)
            return True
        except OSError:
            return False
