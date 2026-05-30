"""
Electroplating Plant Hoist Scheduler
=====================================
Reads:
  input_tanks_csv.csv   – station / tank configuration
  input_wagon_new.csv   – wagon / hoist parameters

Writes:
  OUTPUT_sequence.csv   – PLC instruction sequence
  DIP_TIME_OUTPUT.csv   – per-load dip-time audit log
"""

import math
import pandas as pd
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
INPUT_TANKS  = 'input_tanks_csv.csv'
INPUT_WAGON  = 'input_wagon_new.csv'
OUT_SEQ      = 'OUTPUT_sequence.csv'
OUT_DIP      = 'DIP_TIME_OUTPUT.csv'

TOTAL_LOADS   = 10      # number of loads to process
N_FLIGHTBARS  = 3       # number of FlightBars (hangers/barrels) in this plant
PROJECT_ID    = 'Program 1'
PROGRAM_ID    = 1
WAGON_NO      = 1

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def safe_float(val, default=0.0):
    try:
        x = float(val)
        return default if math.isnan(x) else x
    except (TypeError, ValueError):
        return default

# ─────────────────────────────────────────────────────────────────────────────
# READ INPUTS
# ─────────────────────────────────────────────────────────────────────────────
tanks_df = pd.read_csv(INPUT_TANKS)
wagon_df = pd.read_csv(INPUT_WAGON)
tanks_df.columns = [c.strip() for c in tanks_df.columns]
wagon_df.columns = [c.strip() for c in wagon_df.columns]

SLOW_ZONE_MM = 300.0   # fixed 300 mm slow-approach zone at every station stop

w         = wagon_df.iloc[0]
FAST_SPD  = safe_float(w['Fast Speed Mtrs/Min'])       * 1000 / 60   # mm/s
SUPER_SPD = safe_float(w['Superfast SpeedMtrs/Min'])   * 1000 / 60   # mm/s
SLOW_SPD  = safe_float(w['Slow Speed Mtrs/Min'])       * 1000 / 60   # mm/s  Change 1
LIFT_T    = safe_float(w['Lift Time Seconds'])
LOWER_T   = safe_float(w['Lower Time Seconds'])

# ─────────────────────────────────────────────────────────────────────────────
# BUILD STATION DICTIONARY
# ─────────────────────────────────────────────────────────────────────────────
stations = {}
for _, row in tanks_df.iterrows():
    try:
        sno = int(row['station_no'])
    except (TypeError, ValueError):
        continue
    pno_raw = row['Process_NO']
    active  = pd.notna(pno_raw) and str(pno_raw).strip() not in ('', 'nan')
    mx      = safe_float(row['max_dip_time_sec'])
    can_rest_raw = str(row.get('can_rest_in_return_path', '')).strip().lower()
    stations[sno] = {
        'name'     : str(row['process_name']).strip(),
        'dist'     : safe_float(row['distance_mm']),
        'dip'      : safe_float(row['dip_time_sec']),
        'max_dip'  : mx if mx > 0 else float('inf'),
        'stype'    : str(row['station_type']).strip() if pd.notna(row.get('station_type', '')) else '',
        'active'   : active,
        'pno'      : int(float(pno_raw)) if active else None,
        'can_rest' : can_rest_raw in ('yes', 'true', '1'),
        'Criticality': str(row.get('Criticality', 'LOW')).strip().upper(),
    }

# ─────────────────────────────────────────────────────────────────────────────
# ACTIVE PATH + ALTERNATING TANKS
# ─────────────────────────────────────────────────────────────────────────────
active_snos = sorted(s for s, d in stations.items() if d['active'])
# e.g. [1, 2, 5, 6, 7, 9, 10, 11, 13]

pno_count = Counter(stations[s]['pno'] for s in active_snos)
dup_pnos  = {pno for pno, cnt in pno_count.items() if cnt > 1}
alt_tanks = [s for s in active_snos if stations[s]['pno'] in dup_pnos]
# alt_tanks = [9, 10]  (ZINC PHOSPHATING duplicates)

# Detect LOAD / UNLOAD station from station_type / name columns
LOAD_SNO   = None
UNLOAD_SNO = None
for _sno in active_snos:
    _stype = stations[_sno]['stype'].upper().replace(' ', '')
    _name  = stations[_sno]['name'].upper().replace(' ', '')
    _is_load   = 'LOAD'   in _stype or 'LOAD'   in _name
    _is_unload = 'UNLOAD' in _stype or 'UNLOAD' in _name
    if _is_load and _is_unload:
        LOAD_SNO = UNLOAD_SNO = _sno   # circular plant
        break
    if _is_load   and LOAD_SNO   is None: LOAD_SNO   = _sno
    if _is_unload and UNLOAD_SNO is None: UNLOAD_SNO = _sno
if LOAD_SNO   is None: LOAD_SNO   = active_snos[0]
if UNLOAD_SNO is None: UNLOAD_SNO = active_snos[-1]
is_circular = (LOAD_SNO == UNLOAD_SNO)

# ─────────────────────────────────────────────────────────────────────────────
# TRAVEL TIME  – 3-speed model, single lift OR lower per call   Change 2
# lift=True  -> GET FROM (wagon lifts load at source station)
# lower=True -> PUT ON  (wagon lowers load at destination station)
# ─────────────────────────────────────────────────────────────────────────────
def travel_t(from_sno, to_sno, loaded=True, lift=False, lower=False):
    dist     = abs(stations[to_sno]['dist'] - stations[from_sno]['dist'])
    spd_main = FAST_SPD if loaded else SUPER_SPD
    if spd_main <= 0:
        raise ValueError(
            f"Wagon speed ({'Fast' if loaded else 'Superfast'}) is 0 or missing! Check wagon input."
        )
    slow = SLOW_SPD if SLOW_SPD > 0 else spd_main
    if dist <= SLOW_ZONE_MM:
        traverse = dist / slow
    else:
        traverse = (dist - SLOW_ZONE_MM) / spd_main + SLOW_ZONE_MM / slow
    return traverse + (LIFT_T if lift else 0.0) + (LOWER_T if lower else 0.0)


def crit_rank(sno):
    """Return numeric priority: HIGH=3, MEDIUM=2, LOW=1."""
    c = stations[sno].get('Criticality', 'LOW').upper()
    return {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(c, 1)

# ─────────────────────────────────────────────────────────────────────────────
# CYCLE BOTTLENECK   Change 9: dip/N (not T_ref/N), no inflated uniform_dips
# ─────────────────────────────────────────────────────────────────────────────
pno_to_alts = {}
for s in alt_tanks:
    pno_to_alts.setdefault(stations[s]['pno'], []).append(s)

process_cycle_bottlenecks = []
for pno, tank_list in pno_to_alts.items():
    # throughput bottleneck = dip_time / N_tanks (travel does not scale with N)
    process_cycle_bottlenecks.append(stations[tank_list[0]]['dip'] / len(tank_list))
for sno in active_snos:
    if sno not in alt_tanks:
        process_cycle_bottlenecks.append(stations[sno]['dip'])

target_cycle_time = max(process_cycle_bottlenecks) if process_cycle_bottlenecks else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# EFFECTIVE READY DIP   Change 6: offset = LOWER_T + config dip_time_sec
# Dip clock starts at Lower-End = entry_time + LOWER_T.
# Target lift moment  = entry_time + LOWER_T + dip_time_sec.
# This function returns the offset from entry_time to that target moment.
# ─────────────────────────────────────────────────────────────────────────────
def effective_ready_dip(sno):
    mn = stations[sno]['dip']
    mx = stations[sno]['max_dip']
    target = LOWER_T + mn
    return target if mx == float('inf') else min(target, LOWER_T + mx)

# ─────────────────────────────────────────────────────────────────────────────
# FLIGHTBAR SETUP
# ─────────────────────────────────────────────────────────────────────────────
rest_stations = sorted(
    [s for s, d in stations.items() if d['can_rest']],
    key=lambda s: stations[s]['dist']
)

fb_pool       = list(range(1, N_FLIGHTBARS + 1))
fb_assignment = {}
empty_fbs     = {}   # sno → list[{'id': fb_id, 'avail_time': t}]  Change 8

def _park_empty(sno, fb_id, avail_time=0.0):
    """Append an empty FB to the queue at sno (no overwrite)."""
    empty_fbs.setdefault(sno, []).append({'id': fb_id, 'avail_time': avail_time})

def find_park_station(from_sno):
    d_from = stations[from_sno]['dist']
    d_load = stations[LOAD_SNO]['dist']
    lo, hi = min(d_from, d_load), max(d_from, d_load)
    candidates = [s for s in rest_stations if lo < stations[s]['dist'] < hi]
    if not candidates:
        return None
    return min(candidates, key=lambda s: abs(stations[s]['dist'] - d_from))

# ─────────────────────────────────────────────────────────────────────────────
# DESTINATION LOGIC  (stateless peek + stateful consume)
# ─────────────────────────────────────────────────────────────────────────────
zp_toggle = 0   # which alt tank is next

def peek_dest(from_sno):
    """Return next destination without changing toggle state."""
    idx = active_snos.index(from_sno)

    if from_sno in alt_tanks:                    # leaving an alt-tank group
        pno   = stations[from_sno]['pno']
        group = [s for s in alt_tanks if stations[s]['pno'] == pno]
        last_alt_idx = max(active_snos.index(z) for z in group)
        if last_alt_idx + 1 < len(active_snos):
            nxt = active_snos[last_alt_idx + 1]
        else:
            nxt = UNLOAD_SNO if is_circular else None
    else:
        if idx + 1 < len(active_snos):
            nxt = active_snos[idx + 1]
        else:
            # End of active sequence: return to UNLOAD in circular plant
            nxt = UNLOAD_SNO if (is_circular and from_sno != UNLOAD_SNO) else None

    if nxt is None:
        return None
    if nxt in alt_tanks:                         # entering alt-tank slot
        pno   = stations[nxt]['pno']
        group = sorted([s for s in alt_tanks if stations[s]['pno'] == pno])
        return group[zp_toggle % len(group)]
    return nxt

def consume_dest(from_sno):
    """Return destination AND advance toggle if alt tank was chosen."""
    global zp_toggle
    dest = peek_dest(from_sno)
    if dest in alt_tanks:
        # Find which group this is
        pno = stations[dest]['pno']
        group = sorted([s for s in alt_tanks if stations[s]['pno'] == pno])
        if dest == group[-1]: # if we picked the last one, it doesn't matter, we increment anyway
             pass
        zp_toggle += 1
    return dest

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT ACCUMULATORS
# ─────────────────────────────────────────────────────────────────────────────
seq_rows = []
dip_rows = []
inst_no  = 0
acc_time = 0.0

def add_seq(instruction, value, load_no=0, fb_id=None):
    if load_no == '' or load_no is None: load_no = 0
    # Clean up LOAD_NO if it's a string like '901'
    try:
        load_no = int(load_no)
    except:
        pass
    global inst_no
    inst_no += 1
    seq_rows.append({
        'PROJECT ID'        : PROJECT_ID,
        'Program ID'        : PROGRAM_ID,
        'Wagon Number'      : WAGON_NO,
        'Instruction'       : instruction,
        'Instruction Sr No' : inst_no,
        'Instruction Value' : value,
        'LOAD_NO'           : load_no,
        'FlightBar'         : f'FB{fb_id}' if fb_id is not None else '',
        'ACCUMULATED TIME'  : round(acc_time),
    })

def add_dip(load_id, sno, lower_end_t, lift_start_t):
    # Change 3: dip = Lift-Start minus Lower-End (not PUT ON to pickup)
    # Change 4: target always = config dip_time_sec (never uniform_dips)
    # Change 5: caller passes lower_end_t = entry_time + LOWER_T
    s            = stations[sno]
    actual       = lift_start_t - lower_end_t
    actual_round = round(actual, 1)
    target       = s['dip']
    mn, mx       = s['dip'], s['max_dip']
    if target == 0:
        ok = True
    else:
        ok = mn <= actual_round <= (mx if mx != float('inf') else actual_round + 1)
    dip_rows.append({
        'Load ID'        : load_id,
        'Assigned Tank'  : s['name'],
        'Entry Time (s)' : round(lower_end_t - LOWER_T, 1),
        'Lower End (s)'  : round(lower_end_t, 1),
        'Lift Start (s)' : round(lift_start_t, 1),
        'Exit Time (s)'  : round(lift_start_t, 1),
        'Target Dip (s)' : round(target, 1),
        'Actual Dip (s)' : actual_round,
        'Min Dip (s)'    : round(mn, 1),
        'Max Dip (s)'    : round(mx, 1) if mx != float('inf') else 'inf',
        'Status'         : 'PASS' if ok else 'FAIL',
    })

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE WARMUP  (eliminates startup and end transients)
# ─────────────────────────────────────────────────────────────────────────────
N_WARMUP        = N_FLIGHTBARS
TOTAL_SIM_LOADS = N_WARMUP + TOTAL_LOADS + N_WARMUP

def _make_lid(idx):
    if idx < N_WARMUP:
        return 900 + idx + 1
    if idx < N_WARMUP + TOTAL_LOADS:
        return idx - N_WARMUP + 1
    return 100 + (idx - N_WARMUP - TOTAL_LOADS + 1) # post warmup 101, 102...

REAL_IDS = list(range(1, TOTAL_LOADS + 1))

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
tank_contents  = {}
clock          = 0.0
wagon_pos      = LOAD_SNO
next_load_idx  = 0
unloaded_count = 0

def _try_seed():
    global next_load_idx
    if LOAD_SNO not in tank_contents and fb_pool and next_load_idx < TOTAL_SIM_LOADS:
        new_fb  = fb_pool.pop(0)
        new_lid = _make_lid(next_load_idx)
        fb_assignment[new_lid] = new_fb
        tank_contents[LOAD_SNO] = {
            'load_id': new_lid, 'entry_time': clock, 'fb_id': new_fb
        }
        next_load_idx += 1

_try_seed()   # pre-place first load

for _iteration in range(200_000):
    pending_fbs = sum(len(v) for v in empty_fbs.values())
    if unloaded_count >= TOTAL_SIM_LOADS and pending_fbs == 0:
        break

    # ── Find best LOADED move  Changes 11, 16, 18 ────────────────────────────
    best_sno   = None
    best_score = (float('inf'), float('inf'), float('inf'), float('inf'))

    for sno in sorted(tank_contents.keys(), reverse=True):
        dest = peek_dest(sno)
        if dest is None:
            continue
        if dest in tank_contents:
            continue   # destination occupied — cannot deliver

        lower_end_t   = tank_contents[sno]['entry_time'] + LOWER_T
        target_lift_t = lower_end_t + stations[sno]['dip']
        travel_to_src = travel_t(wagon_pos, sno, loaded=False, lift=True)
        arrive        = clock + travel_to_src
        pickup        = max(arrive, target_lift_t)
        end_t         = pickup + travel_t(sno, dest, loaded=True, lower=True)

        # Chain-free via active_snos index  Change 11
        chain_free_t = end_t
        if dest in active_snos:
            chk_idx = active_snos.index(dest)
            while chk_idx < len(active_snos):
                curr_chk = active_snos[chk_idx]
                if curr_chk == UNLOAD_SNO:
                    break
                c_dip = stations[curr_chk]['dip']
                c_max = stations[curr_chk]['max_dip']
                if c_max != float('inf') and (c_max - c_dip) <= 60:
                    nxt_idx = chk_idx + 1
                    if nxt_idx < len(active_snos):
                        chain_free_t += c_dip + travel_t(
                            curr_chk, active_snos[nxt_idx], loaded=True, lower=True
                        )
                    else:
                        chain_free_t += c_dip
                    chk_idx += 1
                else:
                    break

        violations = 0.0
        for other_sno, other_c in tank_contents.items():
            if other_sno == sno:
                continue
            mx = stations[other_sno]['max_dip']
            if mx == float('inf'):
                continue
            # deadline from Lower-End + max_dip  Change 16
            deadline     = other_c['entry_time'] + LOWER_T + mx
            arrive_after = end_t + travel_t(dest, other_sno, loaded=False, lift=True)
            if arrive_after > deadline:
                violations += (arrive_after - deadline)
            if chain_free_t > deadline:
                violations += (chain_free_t - deadline)

        # Urgency: fraction of max_dip elapsed  Change 18
        mx_own     = stations[sno]['max_dip']
        dip_so_far = max(0.0, clock - lower_end_t)
        urgency    = dip_so_far / mx_own if mx_own not in (float('inf'), 0) else 0.0

        score = (-crit_rank(sno), -urgency, violations, pickup)
        if score < best_score:
            best_score = score
            best_sno   = sno

    # ── Find best EMPTY-FB return ─────────────────────────────────────────────
    best_empty_sno   = None
    best_empty_score = (float('inf'), float('inf'))

    for sno, fb_list in empty_fbs.items():
        if not fb_list:
            continue
        fb_info     = fb_list[0]
        ready_at_fb = fb_info['avail_time']
        arrive_est  = clock + travel_t(wagon_pos, sno, loaded=False, lift=True)
        arrive      = max(arrive_est, ready_at_fb)
        end_t       = arrive + travel_t(sno, LOAD_SNO, loaded=True, lower=True)
        violations  = 0.0
        for other_sno, other_c in tank_contents.items():
            mx = stations[other_sno]['max_dip']
            if mx == float('inf'):
                continue
            deadline     = other_c['entry_time'] + LOWER_T + mx
            arrive_after = end_t + travel_t(LOAD_SNO, other_sno, loaded=False, lift=True)
            if arrive_after > deadline:
                violations += (arrive_after - deadline)
        score = (violations, arrive)
        if score < best_empty_score:
            best_empty_score = score
            best_empty_sno   = sno

    if best_sno is None and best_empty_sno is None:
        if tank_contents:
            soonest = min(
                v['entry_time'] + effective_ready_dip(k)
                for k, v in tank_contents.items()
            )
        else:
            soonest = float('inf')
        soonest_empty = min(
            (fl[0]['avail_time'] for fl in empty_fbs.values() if fl),
            default=float('inf')
        )
        soonest = min(soonest, soonest_empty)
        clock   = max(clock, soonest) + 0.1 if soonest != float('inf') else clock + 0.1
        continue

    # ── Priority decision ─────────────────────────────────────────────────────
    loading_stalled = (
        LOAD_SNO not in tank_contents and not fb_pool
        and pending_fbs > 0 and next_load_idx < TOTAL_LOADS
    )
    empty_viol  = best_empty_score[0]
    loaded_viol = best_score[2] if best_sno is not None else float('inf')
    do_empty = (
        (best_sno is None and best_empty_sno is not None)
        or (loading_stalled and best_empty_sno is not None
            and empty_viol <= loaded_viol)
    )

    if do_empty:
        # ── Empty FlightBar return ────────────────────────────────────────────
        e_sno   = best_empty_sno
        fb_info = empty_fbs[e_sno].pop(0)
        fb_id   = fb_info['id']
        if not empty_fbs[e_sno]:
            del empty_fbs[e_sno]

        ready_at_fb  = fb_info['avail_time']
        travel_to_fb = travel_t(wagon_pos, e_sno, loaded=False, lift=True)
        arrive_empty = clock + travel_to_fb
        wait_for_fb  = max(0.0, ready_at_fb - arrive_empty)
        if wait_for_fb > 0.1:
            acc_time += wait_for_fb
            add_seq('Wait for sec', round(wait_for_fb), '', fb_id)
            clock += wait_for_fb

        # Travel to empty FB – single addition, no double-count  Change 7
        acc_time += travel_to_fb
        clock    += travel_to_fb
        wagon_pos = e_sno
        add_seq('Get from', e_sno, '', fb_id)

        park_sno = find_park_station(e_sno)
        if park_sno is not None:
            tt        = travel_t(e_sno, park_sno, loaded=True, lower=True)
            acc_time += tt
            clock    += tt
            wagon_pos = park_sno
            add_seq('Put on', park_sno, '', fb_id)
            _park_empty(park_sno, fb_id, clock)
        else:
            tt        = travel_t(e_sno, LOAD_SNO, loaded=True, lower=True)
            acc_time += tt
            clock    += tt
            wagon_pos = LOAD_SNO
            add_seq('Put on', LOAD_SNO, '', fb_id)
            fb_pool.append(fb_id)
            _try_seed()

    else:
        # ── Loaded FlightBar move ─────────────────────────────────────────────
        if best_sno is None:
            soonest = min(
                v['entry_time'] + effective_ready_dip(k)
                for k, v in tank_contents.items()
            )
            clock = max(clock, soonest) + 0.1
            continue

        content = tank_contents[best_sno]
        load_id = content['load_id']
        fb_id   = content['fb_id']

        # Backward Wait for Sec – depart exactly when needed  Changes 6 & 7
        lower_end_t   = content['entry_time'] + LOWER_T
        target_lift_t = lower_end_t + stations[best_sno]['dip']
        travel_to_src = travel_t(wagon_pos, best_sno, loaded=False, lift=True)
        required_dep  = target_lift_t - travel_to_src
        wait_secs     = max(0.0, required_dep - clock)

        if wait_secs > 0.1:
            acc_time += wait_secs
            add_seq('Wait for sec', round(wait_secs), load_id, fb_id)
            clock += wait_secs          # advance clock once; no double-count below

        # Travel to source – single addition  Change 7
        acc_time    += travel_to_src
        clock       += travel_to_src
        pickup_time  = clock

        add_seq('Get from', best_sno, load_id, fb_id)

        dest = consume_dest(best_sno)
        tt   = travel_t(best_sno, dest, loaded=True, lower=True)
        acc_time   += tt
        clock      += tt
        entry_time  = clock

        add_seq('Put on', dest, load_id, fb_id)

        # Dip = Lower-End to Lift-Start  Change 5
        old_entry = content['entry_time']
        add_dip(load_id, best_sno, old_entry + LOWER_T, pickup_time)

        del tank_contents[best_sno]
        wagon_pos = dest

        if dest == UNLOAD_SNO:
            if is_circular:
                # Circular: UNLOAD == LOAD; load stays at station for its dip, then seeds next
                dip_req       = stations[UNLOAD_SNO]['dip']
                unloaded_time = entry_time + LOWER_T + dip_req
                add_dip(load_id, dest, entry_time + LOWER_T, unloaded_time)
                unloaded_count += 1
                if next_load_idx < TOTAL_SIM_LOADS:
                    new_lid = _make_lid(next_load_idx)
                    fb_assignment[new_lid] = fb_id
                    tank_contents[LOAD_SNO] = {
                        'load_id': new_lid, 'entry_time': unloaded_time - dip_req, 'fb_id': fb_id
                    }
                    next_load_idx += 1
                else:
                    _park_empty(UNLOAD_SNO, fb_id, unloaded_time)
            else:
                add_dip(load_id, dest, entry_time + LOWER_T, entry_time + LOWER_T)
                unloaded_count += 1
                _park_empty(UNLOAD_SNO, fb_id, entry_time + LOWER_T)
        else:
            tank_contents[dest] = {
                'load_id': load_id, 'entry_time': entry_time, 'fb_id': fb_id
            }

        _try_seed()

# ─────────────────────────────────────────────────────────────────────────────
# WRITE OUTPUTS  (real loads only for dip log; all loads for sequence)
# ─────────────────────────────────────────────────────────────────────────────
dip_real = [r for r in dip_rows if r['Load ID'] in REAL_IDS]

pd.DataFrame(seq_rows).to_csv(OUT_SEQ, index=False)
pd.DataFrame(dip_real).to_csv(OUT_DIP, index=False)

print(f"Done — {unloaded_count}/{TOTAL_SIM_LOADS} simulated loads"
      f" ({N_WARMUP} warmup + {TOTAL_LOADS} real + {N_WARMUP} post-warmup).")
print(f"Target (uniform) cycle time : {target_cycle_time:.1f} s")
print(f"Sequence rows : {len(seq_rows)}")
print(f"Dip log rows  : {len(dip_real)} (real loads only)")
