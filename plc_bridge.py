"""
PLC Bridge for Electroplating Plant Hoist Scheduler
====================================================
Sends the generated instruction sequence to a PLC via Modbus TCP.

REGISTER MAP (Holding Registers, base address 0):
  Python → PLC (write these before setting CMD_EXECUTE):
    HR[0]  INSTRUCTION_TYPE  : 0=Idle, 1=Get_from, 2=Put_on, 3=Wait_for_sec
    HR[1]  INSTRUCTION_VALUE : Station number (1,2,…) or wait seconds
    HR[2]  INSTRUCTION_SR_NO : Instruction serial number (1-based)
    HR[3]  LOAD_NO           : Load number (0 = no load / empty FB move)
    HR[4]  FLIGHTBAR_NO      : 1…N  (0 = not applicable)
    HR[5]  ACCUM_TIME_SEC    : Accumulated time in seconds (scheduler time)
    HR[6]  CMD_EXECUTE       : Python writes 1 here to trigger execution

  PLC → Python (read these):
    HR[10] PLC_READY         : PLC writes 1 when ready for next instruction
    HR[11] CMD_DONE          : PLC writes 1 when current instruction finished
    HR[12] PLC_STATUS        : 0=Idle, 1=Executing, 2=Done, 3=Alarm/Error
    HR[13] PLC_ALARM_CODE    : Non-zero if PLC_STATUS == 3

Install:  pip install pymodbus pandas
"""

import time
import logging
import pandas as pd
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Register addresses (0-based, i.e. HR[0] = address 0) ─────────────────────
R_INSTRUCTION_TYPE  = 0
R_INSTRUCTION_VALUE = 1
R_INSTRUCTION_SR_NO = 2
R_LOAD_NO           = 3
R_FLIGHTBAR_NO      = 4
R_ACCUM_TIME_SEC    = 5
R_CMD_EXECUTE       = 6

R_PLC_READY         = 10
R_CMD_DONE          = 11
R_PLC_STATUS        = 12
R_PLC_ALARM_CODE    = 13

# ── Instruction type encoding ─────────────────────────────────────────────────
INSTRUCTION_CODES = {
    "Get from":    1,
    "Put on":      2,
    "Wait for sec": 3,
}

PLC_STATUS_LABELS = {0: "Idle", 1: "Executing", 2: "Done", 3: "Alarm/Error"}


def parse_fb_number(fb_str: str) -> int:
    """Extract integer from 'FB1', 'FB2', … → 1, 2, … Returns 0 if absent."""
    if not fb_str or not str(fb_str).strip():
        return 0
    s = str(fb_str).strip().upper().replace("FB", "")
    try:
        return int(s)
    except ValueError:
        return 0


def sequence_to_plc_rows(seq_df: pd.DataFrame) -> list[dict]:
    """
    Convert the scheduler output DataFrame to a list of PLC register dicts.
    Rows with unknown instruction types are skipped with a warning.
    """
    rows = []
    for _, row in seq_df.iterrows():
        instr = str(row.get("Instruction", "")).strip()
        code = INSTRUCTION_CODES.get(instr)
        if code is None:
            log.warning("Unknown instruction '%s' at SR#%s — skipped.",
                        instr, row.get("Instruction Sr No", "?"))
            continue

        value = int(row.get("Instruction Value", 0) or 0)
        sr_no = int(row.get("Instruction Sr No", 0) or 0)
        load_no = int(row.get("LOAD_NO", 0) or 0)
        fb_no = parse_fb_number(row.get("FlightBar", ""))
        accum = int(row.get("ACCUMULATED TIME", 0) or 0)

        rows.append({
            "sr_no":       sr_no,
            "instr":       instr,
            "type_code":   code,
            "value":       value,
            "load_no":     load_no,
            "fb_no":       fb_no,
            "accum_time":  accum,
        })
    return rows


class PLCBridge:
    """
    Sends hoist instructions to a PLC one-by-one using a handshake protocol.

    Protocol per instruction
    ────────────────────────
    1. Wait until PLC_READY == 1  (PLC is idle, waiting for command)
    2. Write all register values for the instruction
    3. Write CMD_EXECUTE = 1  (arm)
    4. Wait until CMD_DONE == 1   (PLC finished executing)
    5. Write CMD_EXECUTE = 0  (reset; PLC also resets CMD_DONE)
    6. Move to next instruction
    """

    def __init__(
        self,
        host: str,
        port: int = 502,
        unit_id: int = 1,
        ready_timeout: float = 30.0,
        done_timeout: float = 120.0,
        poll_interval: float = 0.2,
        dry_run: bool = False,
    ):
        """
        Parameters
        ----------
        host            : PLC IP address (e.g. "192.168.1.10")
        port            : Modbus TCP port (default 502)
        unit_id         : Modbus unit/slave ID (default 1)
        ready_timeout   : Max seconds to wait for PLC_READY (per instruction)
        done_timeout    : Max seconds to wait for CMD_DONE  (per instruction)
        poll_interval   : Polling interval in seconds
        dry_run         : If True, print registers without connecting to PLC
        """
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.ready_timeout = ready_timeout
        self.done_timeout = done_timeout
        self.poll_interval = poll_interval
        self.dry_run = dry_run
        self._client: ModbusTcpClient | None = None

    # ── Connection ─────────────────────────────────────────────────────────────
    def connect(self):
        if self.dry_run:
            log.info("[DRY RUN] Skipping PLC connection to %s:%d", self.host, self.port)
            return
        self._client = ModbusTcpClient(host=self.host, port=self.port)
        if not self._client.connect():
            raise ConnectionError(f"Cannot connect to PLC at {self.host}:{self.port}")
        log.info("Connected to PLC at %s:%d (unit %d)", self.host, self.port, self.unit_id)

    def disconnect(self):
        if self._client:
            self._client.close()
            log.info("Disconnected from PLC.")

    # ── Low-level register helpers ─────────────────────────────────────────────
    def _write_reg(self, address: int, value: int):
        if self.dry_run:
            return
        result = self._client.write_register(address, value, slave=self.unit_id)
        if result.isError():
            raise ModbusException(f"Write HR[{address}]={value} failed: {result}")

    def _read_reg(self, address: int) -> int:
        if self.dry_run:
            return 1  # simulate ready/done immediately in dry-run
        result = self._client.read_holding_registers(address, count=1, slave=self.unit_id)
        if result.isError():
            raise ModbusException(f"Read HR[{address}] failed: {result}")
        return result.registers[0]

    def _wait_for_reg(self, address: int, target: int, timeout: float, label: str):
        """Poll register until it equals target or timeout expires."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            val = self._read_reg(address)
            if val == target:
                return
            time.sleep(self.poll_interval)
        raise TimeoutError(
            f"Timeout ({timeout:.0f}s) waiting for {label} "
            f"HR[{address}]=={target} (last read={val})"
        )

    # ── Instruction execution ──────────────────────────────────────────────────
    def _send_instruction(self, reg: dict):
        sr = reg["sr_no"]
        log.info(
            "SR#%03d | %-14s | Value=%-4d | Load=%-3d | FB%d | T=%ds",
            sr, reg["instr"], reg["value"], reg["load_no"], reg["fb_no"], reg["accum_time"],
        )

        if self.dry_run:
            log.info("[DRY RUN] SR#%03d registers written (simulated)", sr)
            return

        # Step 1: wait for PLC to be ready
        self._wait_for_reg(R_PLC_READY, 1, self.ready_timeout, "PLC_READY")

        # Step 2: write instruction registers
        self._write_reg(R_INSTRUCTION_TYPE,  reg["type_code"])
        self._write_reg(R_INSTRUCTION_VALUE, reg["value"])
        self._write_reg(R_INSTRUCTION_SR_NO, reg["sr_no"])
        self._write_reg(R_LOAD_NO,           reg["load_no"])
        self._write_reg(R_FLIGHTBAR_NO,      reg["fb_no"])
        self._write_reg(R_ACCUM_TIME_SEC,    reg["accum_time"])

        # Step 3: trigger execution
        self._write_reg(R_CMD_EXECUTE, 1)

        # Step 4: wait for done
        self._wait_for_reg(R_CMD_DONE, 1, self.done_timeout, "CMD_DONE")

        # Step 5: reset trigger (PLC should mirror this by resetting CMD_DONE)
        self._write_reg(R_CMD_EXECUTE, 0)

        log.info("SR#%03d  DONE", sr)

    # ── Public API ─────────────────────────────────────────────────────────────
    def run_sequence(self, seq_df: pd.DataFrame, start_from_sr: int = 1):
        """
        Execute all instructions in seq_df.

        Parameters
        ----------
        seq_df       : DataFrame from run_scheduler() (sequence output)
        start_from_sr: Resume from this instruction serial number (for restart)
        """
        instructions = sequence_to_plc_rows(seq_df)
        total = len(instructions)
        log.info("Loaded %d instructions. Starting from SR#%d.", total, start_from_sr)

        executed = 0
        for reg in instructions:
            if reg["sr_no"] < start_from_sr:
                continue

            # Check PLC alarm before each instruction
            if not self.dry_run:
                status = self._read_reg(R_PLC_STATUS)
                if status == 3:
                    alarm = self._read_reg(R_PLC_ALARM_CODE)
                    raise RuntimeError(
                        f"PLC alarm before SR#{reg['sr_no']}! "
                        f"Alarm code = {alarm}. Sequence halted."
                    )

            self._send_instruction(reg)
            executed += 1

        log.info("Sequence complete. %d instructions executed.", executed)


# ─────────────────────────────────────────────────────────────────────────────
# ALTERNATIVE PROTOCOLS
# ─────────────────────────────────────────────────────────────────────────────
# If your PLC does not support Modbus TCP, use one of these instead:
#
# ── Siemens S7-300/400/1200/1500 (Snap7) ─────────────────────────────────────
#   pip install python-snap7
#
#   import snap7
#   plc = snap7.client.Client()
#   plc.connect("192.168.1.10", 0, 1)           # ip, rack, slot
#   # Write DB1.DBW0 (Data Block 1, Word offset 0):
#   data = snap7.util.set_int(bytearray(2), 0, value)
#   plc.db_write(db_number=1, start=0, data=data)
#
# ── Allen-Bradley / Rockwell (EtherNet/IP) ────────────────────────────────────
#   pip install pycomm3
#
#   from pycomm3 import LogixDriver
#   with LogixDriver("192.168.1.10") as plc:
#       plc.write(("INSTRUCTION_TYPE", instr_code))
#       plc.write(("INSTRUCTION_VALUE", value))
#       plc.write(("CMD_EXECUTE", 1))
#
# ── OPC-UA (vendor-neutral, works with Siemens/Beckhoff/Schneider/etc.) ───────
#   pip install asyncua
#
#   from asyncua.sync import Client
#   with Client("opc.tcp://192.168.1.10:4840") as client:
#       node = client.get_node("ns=3;s=INSTRUCTION_TYPE")
#       node.write_value(instr_code)
#
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE USAGE  (run directly: python plc_bridge.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    PLC_IP   = "192.168.1.10"   # <-- change to your PLC IP
    CSV_PATH = "OUTPUT_sequence.csv"  # generated by the Streamlit app

    if len(sys.argv) > 1:
        PLC_IP = sys.argv[1]
    if len(sys.argv) > 2:
        CSV_PATH = sys.argv[2]

    seq_df = pd.read_csv(CSV_PATH)
    log.info("Loaded sequence from '%s' (%d rows)", CSV_PATH, len(seq_df))

    bridge = PLCBridge(
        host=PLC_IP,
        port=502,
        unit_id=1,
        ready_timeout=30.0,
        done_timeout=120.0,
        dry_run=False,      # set True to test without a real PLC
    )

    try:
        bridge.connect()
        bridge.run_sequence(seq_df, start_from_sr=1)
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
    except Exception as exc:
        log.error("Fatal: %s", exc)
        raise
    finally:
        bridge.disconnect()
