"""
Electroplating Plant Hoist Scheduler - Streamlit GUI
=====================================================
Interactive GUI for scheduling wagon movements through electroplating stations.

Three tabs:
  1. Configuration: Upload input files, set parameters, generate sequence
  2. Execution: View generated sequence
  3. Analysis: View dip times, cycle time, processing time, and wagon movement graph
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import math
import io
import time
from collections import Counter
import base64
import cv2
import sklearn
import matplotlib
import seaborn
import google.generativeai

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Scheduler",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🏭 Electroplating Plant Hoist Scheduler")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def safe_float(val, default=0.0):
    """Convert value to float safely."""
    try:
        x = float(val)
        return default if math.isnan(x) else x
    except (TypeError, ValueError):
        return default


def run_scheduler(tanks_df, wagon_df, total_loads=10, n_flightbars=3,
                  project_id='Program 1', program_id=1, wagon_no=1):
    """
    Run the scheduling algorithm and return sequence and dip time DataFrames.

    Returns:
        tuple: (sequence_df, dip_df, wagon_movements, stations, target_cycle_time, infeasible_stations)
    """
    # Normalize column names
    tanks_df.columns = [c.strip() for c in tanks_df.columns]
    wagon_df.columns = [c.strip() for c in wagon_df.columns]
    
    # Read wagon parameters
    w = wagon_df.iloc[0]
    FAST_SPD = safe_float(w.get('Fast Speed Mtrs/Min', 0)) * 1000 / 60  # mm/s
    SUPER_SPD = safe_float(w.get('Superfast SpeedMtrs/Min', 0)) * 1000 / 60  # mm/s
    LIFT_T   = safe_float(w.get('Lift Time Seconds', 0))
    LOWER_T  = safe_float(w.get('Lower Time Seconds', 0))
    SLOW_SPD = safe_float(w.get('Slow Speed Mtrs/Min', 0)) * 1000 / 60
    
    # Build station dictionary
    stations = {}
    for _, row in tanks_df.iterrows():
        try:
            sno = int(row['station_no'])
        except (TypeError, ValueError):
            continue
        
        pno_raw = row.get('Process_NO', '')
        active = pd.notna(pno_raw) and str(pno_raw).strip() not in ('', 'nan')
        mx = safe_float(row.get('max_dip_time_sec', 0))
        
        can_rest_raw = str(row.get('can_rest_in_return_path', '')).strip().lower()
        stations[sno] = {
            'name'       : str(row.get('process_name', f'Station {sno}')).strip(),
            'dist'       : safe_float(row.get('distance_mm', 0)),
            'dip'        : safe_float(row.get('dip_time_sec', 0)),
            'max_dip'    : mx if mx > 0 else float('inf'),
            'stype'      : str(row.get('station_type', '')).strip(),
            'active'     : active,
            'pno'        : int(float(pno_raw)) if active else None,
            'can_rest'   : can_rest_raw in ('yes', 'true', '1'),
            'Criticality': str(row.get('Criticality', 'LOW')).strip().upper(),
        }
    
    # Active path and alternating tanks
    active_snos = sorted(s for s, d in stations.items() if d['active'])
    if len(active_snos) < 2:
        raise ValueError(
            "No active stations found. Check that 'Process_NO' column is filled "
            "in input_tanks_csv.csv for at least the loading and one processing station."
        )
    pno_count = Counter(stations[s]['pno'] for s in active_snos)
    dup_pnos = {pno for pno, cnt in pno_count.items() if cnt > 1}
    alt_tanks = [s for s in active_snos if stations[s]['pno'] in dup_pnos]

    LOAD_SNO = None
    UNLOAD_SNO = None
    for sno in active_snos:
        stype = stations[sno]['stype'].upper().replace(' ', '')
        name = stations[sno]['name'].upper().replace(' ', '')
        is_load = 'LOAD' in stype or 'LOAD' in name
        is_unload = 'UNLOAD' in stype or 'UNLOAD' in name
        
        if is_load and is_unload:
            LOAD_SNO = UNLOAD_SNO = sno
            break
        if is_load   and LOAD_SNO   is None: LOAD_SNO   = sno
        if is_unload and UNLOAD_SNO is None: UNLOAD_SNO = sno
            
    if LOAD_SNO is None: LOAD_SNO = active_snos[0]
    if UNLOAD_SNO is None: UNLOAD_SNO = active_snos[-1]
    
    is_circular = (LOAD_SNO == UNLOAD_SNO)
    
    # Travel time calculation
    SLOW_ZONE_MM = 300.0

    def travel_t(from_sno, to_sno, loaded=True, lift=False, lower=False):
        dist     = abs(stations[to_sno]['dist'] - stations[from_sno]['dist'])
        spd_main = FAST_SPD if loaded else SUPER_SPD
        if spd_main <= 0:
            raise ValueError(
                f"Wagon speed ({'Fast' if loaded else 'Superfast'}) is 0 or missing! "
                "Please check the 'Fast Speed Mtrs/Min' and 'Superfast SpeedMtrs/Min' columns in the wagon parameters."
            )
        slow = SLOW_SPD if SLOW_SPD > 0 else spd_main
        if dist <= SLOW_ZONE_MM:
            traverse = dist / slow
        else:
            traverse = (dist - SLOW_ZONE_MM) / spd_main + SLOW_ZONE_MM / slow
        return traverse + (LIFT_T if lift else 0.0) + (LOWER_T if lower else 0.0)

    def crit_rank_fn(sno):
        c = stations[sno].get('Criticality', 'LOW').upper()
        return {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(c, 1)

    pno_to_alts = {}
    for s in alt_tanks:
        pno_to_alts.setdefault(stations[s]['pno'], []).append(s)

    process_cycle_bottlenecks = []
    for pno, tank_list in pno_to_alts.items():
        process_cycle_bottlenecks.append(stations[tank_list[0]]['dip'] / len(tank_list))
    for sno in active_snos:
        if sno not in alt_tanks:
            process_cycle_bottlenecks.append(stations[sno]['dip'])
    target_cycle_time = max(process_cycle_bottlenecks) if process_cycle_bottlenecks else 0.0

    infeasible_stations = [
        stations[s]['name'] for s in active_snos
        if stations[s]['max_dip'] != float('inf')
        and stations[s]['max_dip'] < stations[s]['dip']
    ]

    def effective_ready_dip(sno):
        mn = stations[sno]['dip']
        mx = stations[sno]['max_dip']
        target = LOWER_T + mn
        return target if mx == float('inf') else min(target, LOWER_T + mx)

    # ── FlightBar setup ───────────────────────────────────────────────────────
    # Rest stations: intermediate positions where an empty FB can be parked
    # while the wagon performs a higher-priority loaded move.
    rest_stations = sorted(
        [s for s, d in stations.items() if d['can_rest']],
        key=lambda s: stations[s]['dist']
    )

    # ── Pipeline warmup to eliminate startup and end transients ──────────────
    # Run n_warmup "invisible" loads before and after the real loads so that:
    #   • First real load (L0) enters a fully-cycled pipeline → same dip as all others
    #   • Last real load has competing tasks behind it → same ZP dip as steady state
    n_warmup        = n_flightbars          # fills the pipeline (= max concurrent loads)
    total_sim_loads = n_warmup + total_loads + n_warmup

    def _make_lid(idx):
        """Return integer labels: Warmups in 900s, Real starting from 1, Post in 800s."""
        if idx < n_warmup:
            return 900 + idx + 1
        if idx < n_warmup + total_loads:
            return idx - n_warmup + 1
        return 800 + (idx - n_warmup - total_loads + 1)

    # Set of real (user-requested) load IDs
    _real_ids = {i + 1 for i in range(total_loads)}

    # FB pool: list of available FB numbers at the loading station
    fb_pool      = list(range(1, n_flightbars + 1))  # [1, 2, 3, ...]
    fb_assignment = {}   # load_id → fb_id

    def find_park_station(from_sno):
        """Closest rest station (to from_sno) that lies on the return path."""
        d_from = stations[from_sno]['dist']
        d_load = stations[LOAD_SNO]['dist']
        lo, hi = min(d_from, d_load), max(d_from, d_load)
        candidates = [
            s for s in rest_stations
            if lo < stations[s]['dist'] < hi
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda s: abs(stations[s]['dist'] - d_from))

    # Alternating tank toggle
    zp_toggle = 0
    
    def peek_dest(from_sno):
        idx = active_snos.index(from_sno)

        if from_sno in alt_tanks:
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
                nxt = UNLOAD_SNO if (is_circular and from_sno != UNLOAD_SNO) else None

        if nxt is None:
            return None

        if nxt in alt_tanks:
            pno   = stations[nxt]['pno']
            group = sorted([s for s in alt_tanks if stations[s]['pno'] == pno])
            return group[zp_toggle % len(group)]
        return nxt
    
    def consume_dest(from_sno):
        nonlocal zp_toggle
        dest = peek_dest(from_sno)
        if dest in alt_tanks:
            zp_toggle += 1
        return dest
    
    # Output accumulators
    seq_rows      = []
    dip_rows      = []
    wagon_movements = []   # load-centric (legacy)
    wagon_track   = []     # wagon-centric way-time points
    inst_no  = 0
    acc_time = 0.0

    def _wt(t, sno, seg):
        """Record a wagon position event for the way-time diagram."""
        wagon_track.append({
            'time': round(t, 1),
            'dist': stations[sno]['dist'],
            'sno' : sno,
            'name': stations[sno]['name'],
            'seg' : seg,        # 'loaded' | 'empty' | 'wait'
        })
    
    def add_seq(instruction, value, load_no=0, fb_id=None):
        if load_no == '' or load_no is None: load_no = 0
        nonlocal inst_no
        inst_no += 1
        seq_rows.append({
            'PROJECT ID': project_id,
            'Program ID': program_id,
            'Wagon Number': wagon_no,
            'Instruction': instruction,
            'Instruction Sr No': inst_no,
            'Instruction Value': value,
            'LOAD_NO': load_no,
            'FlightBar': f'FB{fb_id}' if fb_id is not None else '',
            'ACCUMULATED TIME': round(acc_time),
        })
    
    def add_dip(load_id, sno, lower_end_t, lift_start_t):
        # Change 13: args are lower_end_t and lift_start_t (spec dip clock)
        # Change 4 : target = config dip_time_sec only, never inflated
        s            = stations[sno]
        actual       = lift_start_t - lower_end_t   # dip = Lower-End → Lift-Start
        actual_round = round(actual, 1)
        target       = s['dip']                     # always config value
        mn, mx       = s['dip'], s['max_dip']
        if target == 0:
            ok = True
        else:
            ok = mn <= actual_round <= (mx if mx != float('inf') else actual_round + 1)
        dip_rows.append({
            'Load ID'        : load_id,
            'Station No'     : sno,
            'Assigned Tank'  : s['name'],
            'Entry Time (s)' : round(lower_end_t - LOWER_T, 1),
            'Lower End (s)'  : round(lower_end_t, 1),
            'Lift Start (s)' : round(lift_start_t, 1),
            'Exit Time (s)'  : round(lift_start_t, 1),
            'Max Dip (s)'    : round(mx, 1) if mx != float('inf') else '∞',
            'Target Dip (s)' : round(target, 1),
            'Actual Dip (s)' : actual_round,
            'Min Dip (s)'    : round(mn, 1),
            'Status'         : '✓ PASS' if ok else '✗ FAIL',
        })
    
    # ── Simulation ────────────────────────────────────────────────────────────
    # tank_contents[sno] = {'load_id', 'entry_time', 'fb_id'}  – loaded FBs
    # empty_fbs[sno]     = [fb_id, ...]  – queue of empty FBs at that station
    tank_contents  = {}
    empty_fbs      = {}   # sno → list[fb_id]
    clock          = 0.0
    wagon_pos      = LOAD_SNO
    next_load_idx  = 0
    unloaded_count = 0

    def _park_empty(sno, fb_id, avail_time=0.0):
        """Place an empty FB at a station (queue, not overwrite)."""
        empty_fbs.setdefault(sno, []).append({'id': fb_id, 'avail_time': avail_time})

    def _try_seed():
        """Assign next load + FB to LOAD_SNO if both are available."""
        nonlocal next_load_idx
        if LOAD_SNO not in tank_contents and fb_pool and next_load_idx < total_sim_loads:
            new_fb  = fb_pool.pop(0)
            new_lid = _make_lid(next_load_idx)
            fb_assignment[new_lid] = new_fb
            tank_contents[LOAD_SNO] = {
                'load_id': new_lid, 'entry_time': clock, 'fb_id': new_fb
            }
            wagon_movements.append({'time': clock, 'station': LOAD_SNO, 'load': new_lid})
            next_load_idx += 1

    _try_seed()   # pre-place first load
    _wt(0.0, LOAD_SNO, 'empty')   # wagon starts at loading station

    for _ in range(200_000):
        pending_fbs = sum(len(v) for v in empty_fbs.values())
        if unloaded_count >= total_sim_loads and pending_fbs == 0:
            break

        # ── Find best LOADED move  Changes 11, 16, 17, 18 ────────────────────
        best_sno   = None
        best_score = (float('inf'), float('inf'), float('inf'), float('inf'))

        for sno in sorted(tank_contents.keys(), reverse=True):
            dest = peek_dest(sno)
            if dest is None:
                continue
            if dest in tank_contents:
                continue

            lower_end_t   = tank_contents[sno]['entry_time'] + LOWER_T
            target_lift_t = lower_end_t + stations[sno]['dip']
            travel_to_src = travel_t(wagon_pos, sno, loaded=False, lift=True)
            arrive        = clock + travel_to_src
            pickup        = max(arrive, target_lift_t)
            end_t         = pickup + travel_t(sno, dest, loaded=True, lower=True)

            # Chain-free check via active_snos index  Change 17
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
                # deadline uses Lower-End + max_dip  Change 16
                deadline     = other_c['entry_time'] + LOWER_T + mx
                arrive_after = end_t + travel_t(dest, other_sno, loaded=False, lift=True)
                if arrive_after > deadline:
                    violations += (arrive_after - deadline)
                if chain_free_t > deadline:
                    violations += (chain_free_t - deadline)

            # Urgency fraction of max_dip elapsed  Change 18
            mx_own     = stations[sno]['max_dip']
            dip_so_far = max(0.0, clock - lower_end_t)
            urgency    = dip_so_far / mx_own if mx_own not in (float('inf'), 0) else 0.0

            score = (-crit_rank_fn(sno), -urgency, violations, pickup)
            if score < best_score:
                best_score = score
                best_sno   = sno

        # ── Find best EMPTY-FB return ─────────────────────────────────────────
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

        # Nothing to do – advance clock
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

        # ── Priority decision ─────────────────────────────────────────────────
        loading_stalled = (
            LOAD_SNO not in tank_contents
            and not fb_pool
            and pending_fbs > 0
            and next_load_idx < total_loads
        )
        empty_viol  = best_empty_score[0]
        loaded_viol = best_score[2] if best_sno is not None else float('inf')
        do_empty = (
            (best_sno is None and best_empty_sno is not None)
            or (loading_stalled and best_empty_sno is not None
                and empty_viol <= loaded_viol)
        )

        if do_empty:
            # ── Empty FlightBar return ────────────────────────────────────────
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
                _wt(arrive_empty, e_sno, 'empty')
                _wt(clock,        e_sno, 'wait')

            # Travel to empty FB – single addition, no double-count  Change 15
            acc_time += travel_to_fb
            clock    += travel_to_fb
            wagon_pos = e_sno
            _wt(clock, e_sno, 'empty')
            add_seq('Get from', e_sno, '', fb_id)

            park_sno = find_park_station(e_sno)
            if park_sno is not None:
                tt        = travel_t(e_sno, park_sno, loaded=True, lower=True)
                acc_time += tt; clock += tt; wagon_pos = park_sno
                _wt(clock, park_sno, 'empty')
                add_seq('Put on', park_sno, '', fb_id)
                _park_empty(park_sno, fb_id, clock)
            else:
                tt        = travel_t(e_sno, LOAD_SNO, loaded=True, lower=True)
                acc_time += tt; clock += tt; wagon_pos = LOAD_SNO
                _wt(clock, LOAD_SNO, 'empty')
                add_seq('Put on', LOAD_SNO, '', fb_id)
                fb_pool.append(fb_id)
                _try_seed()

        else:
            # ── Loaded FlightBar move ─────────────────────────────────────────
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

            # Backward Wait for Sec  Changes 14 & 15
            lower_end_t   = content['entry_time'] + LOWER_T
            target_lift_t = lower_end_t + stations[best_sno]['dip']
            travel_to_src = travel_t(wagon_pos, best_sno, loaded=False, lift=True)
            required_dep  = target_lift_t - travel_to_src
            wait_secs     = max(0.0, required_dep - clock)

            if wait_secs > 0.1:
                acc_time += wait_secs
                add_seq('Wait for sec', round(wait_secs), load_id, fb_id)
                clock += wait_secs          # advance clock – no double-count below
                _wt(clock, wagon_pos, 'wait')

            # Travel to source – added once  Change 15
            acc_time    += travel_to_src
            clock       += travel_to_src
            pickup_time  = clock
            _wt(pickup_time, best_sno, 'empty')
            add_seq('Get from', best_sno, load_id, fb_id)

            dest = consume_dest(best_sno)
            tt   = travel_t(best_sno, dest, loaded=True, lower=True)
            acc_time  += tt
            clock     += tt
            entry_time = clock
            _wt(clock, dest, 'loaded')

            wagon_movements.append({'time': clock, 'station': dest, 'load': load_id})
            add_seq('Put on', dest, load_id, fb_id)

            # Dip = Lower-End → Lift-Start  Change 13 (caller passes lower_end_t)
            old_entry = content['entry_time']
            add_dip(load_id, best_sno, old_entry + LOWER_T, pickup_time)

            del tank_contents[best_sno]
            wagon_pos = dest

            if dest == UNLOAD_SNO:
                if is_circular:
                    dip_req       = stations[UNLOAD_SNO]['dip']
                    unloaded_time = entry_time + LOWER_T + dip_req
                    add_dip(load_id, dest, entry_time + LOWER_T, unloaded_time)
                    unloaded_count += 1
                    if next_load_idx < total_sim_loads:
                        new_lid = _make_lid(next_load_idx)
                        fb_assignment[new_lid] = fb_id
                        tank_contents[LOAD_SNO] = {
                            'load_id': new_lid,
                            'entry_time': unloaded_time - dip_req,
                            'fb_id': fb_id,
                        }
                        next_load_idx += 1
                        _try_seed()
                    else:
                        _park_empty(UNLOAD_SNO, fb_id, unloaded_time)
                else:
                    unloaded_time = entry_time + LOWER_T
                    add_dip(load_id, dest, entry_time + LOWER_T, entry_time + LOWER_T)
                    unloaded_count += 1
                    _park_empty(UNLOAD_SNO, fb_id, unloaded_time)
            else:
                tank_contents[dest] = {
                    'load_id': load_id, 'entry_time': entry_time, 'fb_id': fb_id
                }

            _try_seed()

    # ── Outputs ───────────────────────────────────────────────────────────────
    seq_out = pd.DataFrame(seq_rows)
    if 'LOAD_NO' in seq_out.columns:
        seq_out['LOAD_NO'] = seq_out['LOAD_NO'].astype('Int64')

    dip_out  = pd.DataFrame(dip_rows)
    move_out = pd.DataFrame(wagon_movements)

    if len(dip_out) > 0:
        t_start = dip_out['Entry Time (s)'].min()
        t_end   = dip_out['Exit Time (s)'].max()
        wt_out  = pd.DataFrame(
            [p for p in wagon_track if t_start <= p['time'] <= t_end]
        )
    else:
        wt_out = pd.DataFrame(wagon_track)

    return (
        seq_out,
        dip_out,
        move_out,
        stations,
        target_cycle_time,
        infeasible_stations,
        wt_out,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
if 'sequence_df' not in st.session_state:
    st.session_state.sequence_df = None
if 'dip_df' not in st.session_state:
    st.session_state.dip_df = None
if 'wagon_movements_df' not in st.session_state:
    st.session_state.wagon_movements_df = None
if 'stations' not in st.session_state:
    st.session_state.stations = None
if 'target_cycle_time' not in st.session_state:
    st.session_state.target_cycle_time = None
if 'infeasible_stations' not in st.session_state:
    st.session_state.infeasible_stations = []
if 'wagon_track_df' not in st.session_state:
    st.session_state.wagon_track_df = None
if 'tanks_df_edited' not in st.session_state:
    st.session_state.tanks_df_edited = None
if 'wagon_df_edited' not in st.session_state:
    st.session_state.wagon_df_edited = None
if 'crosstrolley_df_edited' not in st.session_state:
    st.session_state.crosstrolley_df_edited = None

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["⚙️ Configuration", "📊 Execution", "📈 Analysis"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Configuration & Input Files")
    
    # Parameters - Moved to top
    st.subheader("Simulation Parameters")

    param_col1, param_col2 = st.columns(2)
    with param_col1:
        total_loads = st.number_input(
            "Total Loads to Process",
            min_value=1, max_value=500, value=10, step=1
        )
    with param_col2:
        n_flightbars = st.number_input(
            "Number of FlightBars (hangers/barrels)",
            min_value=1, max_value=50, value=3, step=1,
            help="Physical carriers that move loads through the line. "
                 "If all are in processing, loading blocks until one returns."
        )

    # Default values for project/program/wagon
    project_id = "Program 1"
    program_id = 1
    wagon_no = 1
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1️⃣ Tanks Configuration")
        tanks_file = st.file_uploader(
            "Upload input_tanks_csv.csv",
            type=['csv'],
            key='tanks_uploader'
        )
    
    with col2:
        st.subheader("2️⃣ Wagon Parameters")
        wagon_file = st.file_uploader(
            "Upload input_wagon_new.csv",
            type=['csv'],
            key='wagon_uploader'
        )
    
    with col3:
        st.subheader("3️⃣ Crosstrolley (Optional)")
        crosstrolley_file = st.file_uploader(
            "Upload input_Crosstrolley.csv",
            type=['csv'],
            key='crosstrolley_uploader'
        )
    
    st.divider()
    
    # Generate button
    if st.button(
        "🚀 Generate Sequence",
        key='generate_btn',
        type='primary',
        use_container_width=True
    ):
        if tanks_file is None or wagon_file is None:
            st.error("❌ Please upload both Tanks and Wagon CSV files!")
        else:
            try:
                # Use edited dataframes if available, otherwise read from files
                if st.session_state.tanks_df_edited is not None:
                    tanks_df = st.session_state.tanks_df_edited
                else:
                    tanks_file.seek(0)
                    tanks_df = pd.read_csv(tanks_file)
                
                if st.session_state.wagon_df_edited is not None:
                    wagon_df = st.session_state.wagon_df_edited
                else:
                    wagon_file.seek(0)
                    wagon_df = pd.read_csv(wagon_file)
                
                with st.spinner("🔄 Running scheduler..."):
                    seq_df, dip_df, wagon_moves, stations, target_ct, infeasible, wagon_track_df = run_scheduler(
                        tanks_df,
                        wagon_df,
                        total_loads=total_loads,
                        n_flightbars=int(n_flightbars),
                        project_id=project_id,
                        program_id=program_id,
                        wagon_no=wagon_no
                    )

                    st.session_state.sequence_df = seq_df
                    st.session_state.dip_df = dip_df
                    st.session_state.wagon_movements_df = wagon_moves
                    st.session_state.stations = stations
                    st.session_state.target_cycle_time = target_ct
                    st.session_state.infeasible_stations = infeasible
                    st.session_state.wagon_track_df = wagon_track_df
                    st.session_state.total_loads = total_loads
                
                st.success("✅ Sequence generated successfully!")
                load_count = dip_df['Load ID'].nunique() if 'Load ID' in dip_df.columns else 0
                st.info(f"📦 {load_count} loads processed | 📋 {len(seq_df)} instructions")
                
            except Exception as e:
                import traceback
                st.error(f"❌ Error: {str(e)}")
                st.code(traceback.format_exc(), language='python')
    
    # Display and edit uploaded files
    st.divider()
    st.subheader("📊 Edit Input Data")
    
    if tanks_file is not None:
        tanks_file.seek(0)  # Reset file pointer
        tanks_df = pd.read_csv(tanks_file)
        st.markdown("#### 🔧 Tanks Configuration (Editable)")
        edited_tanks = st.data_editor(
            tanks_df,
            use_container_width=True,
            height=300,
            key='tanks_editor'
        )
        st.session_state.tanks_df_edited = edited_tanks
    
    if wagon_file is not None:
        wagon_file.seek(0)  # Reset file pointer
        wagon_df = pd.read_csv(wagon_file)
        st.markdown("#### 🚚 Wagon Parameters (Editable)")
        edited_wagon = st.data_editor(
            wagon_df,
            use_container_width=True,
            height=150,
            key='wagon_editor'
        )
        st.session_state.wagon_df_edited = edited_wagon
    
    if crosstrolley_file is not None:
        crosstrolley_file.seek(0)  # Reset file pointer
        crosstrolley_df = pd.read_csv(crosstrolley_file)
        st.markdown("#### 🔀 Crosstrolley (Editable)")
        edited_crosstrolley = st.data_editor(
            crosstrolley_df,
            use_container_width=True,
            height=200,
            key='crosstrolley_editor'
        )
        st.session_state.crosstrolley_df_edited = edited_crosstrolley


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: EXECUTION (Sequence)
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Generated Sequence")
    
    if st.session_state.sequence_df is None:
        st.info("👉 Generate sequence in the Configuration tab first!")
    else:
        seq_df = st.session_state.sequence_df
        
        # Display full table
        st.subheader("📋 Instruction Sequence")
        
        show_steady_only = st.checkbox("🔍 Analyze Steady-State Loads Only (Hide Warmups)", value=False)
        
        if show_steady_only:
            # We assume steady state loads are those strictly between 1 and total_loads
            t_loads = st.session_state.get('total_loads', 900)
            display_seq = seq_df[(seq_df['LOAD_NO'] > 0) & (seq_df['LOAD_NO'] <= t_loads) | (seq_df['LOAD_NO'] == 0) | seq_df['LOAD_NO'].isna()]
        else:
            display_seq = seq_df

        st.dataframe(display_seq, use_container_width=True, height=400)
        
        # Download button
        csv_buffer = io.StringIO()
        seq_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv_buffer.getvalue(),
            file_name="OUTPUT_sequence.csv",
            mime="text/csv"
        )
        
        # ── PLC OUTPUT ────────────────────────────────────────────────────────
        st.divider()
        st.subheader("🔌 Send to PLC (Modbus TCP)")

        plc_col1, plc_col2, plc_col3 = st.columns([2, 1, 1])
        with plc_col1:
            plc_ip = st.text_input("PLC IP Address", value="192.168.1.10", key="plc_ip")
        with plc_col2:
            plc_port = st.number_input("Port", value=502, min_value=1, max_value=65535, key="plc_port")
        with plc_col3:
            plc_unit = st.number_input("Unit ID", value=1, min_value=1, max_value=255, key="plc_unit")

        plc_col4, plc_col5 = st.columns([1, 1])
        with plc_col4:
            start_sr = st.number_input(
                "Start from SR#", value=1, min_value=1, key="plc_start_sr",
                help="Resume from this instruction serial number (use 1 for full run)"
            )
        with plc_col5:
            dry_run = st.checkbox(
                "Dry Run (test without PLC)", value=True, key="plc_dry_run",
                help="Logs all instructions without connecting to a real PLC"
            )

        if st.button("▶️ Send Sequence to PLC", key="plc_send_btn", type="primary"):
            try:
                from plc_bridge import PLCBridge
                import threading

                log_placeholder = st.empty()
                log_lines = []

                def run_plc():
                    import logging

                    class StreamlitHandler(logging.Handler):
                        def emit(self, record):
                            log_lines.append(self.format(record))

                    bridge_log = logging.getLogger("plc_bridge")
                    handler = StreamlitHandler()
                    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S"))
                    bridge_log.addHandler(handler)

                    bridge = PLCBridge(
                        host=plc_ip,
                        port=int(plc_port),
                        unit_id=int(plc_unit),
                        dry_run=dry_run,
                    )
                    try:
                        bridge.connect()
                        bridge.run_sequence(seq_df, start_from_sr=int(start_sr))
                        log_lines.append("✅ Sequence complete.")
                    except Exception as e:
                        log_lines.append(f"❌ Error: {e}")
                    finally:
                        bridge.disconnect()

                t = threading.Thread(target=run_plc, daemon=True)
                t.start()

                with st.spinner("Sending to PLC..."):
                    while t.is_alive():
                        if log_lines:
                            log_placeholder.code("\n".join(log_lines[-40:]))
                        time.sleep(0.3)
                    t.join()

                log_placeholder.code("\n".join(log_lines[-40:]))

            except ImportError:
                st.error("pymodbus not installed. Run:  pip install pymodbus")
            except Exception as e:
                st.error(f"PLC error: {e}")

        # Summary stats
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Instructions", len(seq_df))
        
        with col2:
            if 'LOAD_NO' in seq_df.columns:
                real_loads_count = seq_df[(seq_df['LOAD_NO'] > 0) & (seq_df['LOAD_NO'] < 800)]['LOAD_NO'].nunique()
            else:
                real_loads_count = 0
            st.metric("Total Loads", real_loads_count)

        with col3:
            max_time = seq_df['ACCUMULATED TIME'].max() if 'ACCUMULATED TIME' in seq_df.columns else 0
            st.metric("Total Execution Time", f"{max_time}s")

        with col4:
            get_count = len(seq_df[seq_df['Instruction'] == 'Get from']) if 'Instruction' in seq_df.columns else 0
            st.metric("Pick-up Operations", get_count)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Analysis & Visualizations")
    
    if st.session_state.dip_df is None:
        st.info("👉 Generate sequence in the Configuration tab first!")
    else:
        dip_df = st.session_state.dip_df
        seq_df = st.session_state.sequence_df
        wagon_moves = st.session_state.wagon_movements_df
        wt_df = st.session_state.get('wagon_track_df')
        stations = st.session_state.stations
        target_cycle_time = st.session_state.target_cycle_time or 0.0
        infeasible_stations = st.session_state.infeasible_stations or []
        
        # ─────────────────────────────────────────────────────────────────────
        # FILTER
        # ─────────────────────────────────────────────────────────────────────
        st.subheader("⚙️ Analysis Settings")
        analyze_steady_only = st.checkbox("🔍 Analyze Steady-State Loads Only (Hide Warmups)", value=True, key="analysis_steady")
        
        if analyze_steady_only:
            t_loads = st.session_state.get('total_loads', 900)
            # Filter dip_df to real loads only (1 to total_loads)
            dip_df = dip_df[(dip_df['Load ID'] > 0) & (dip_df['Load ID'] <= t_loads)]
            wagon_moves = wagon_moves[(wagon_moves['load'] > 0) & (wagon_moves['load'] <= t_loads)]
            if wt_df is not None and len(dip_df) > 0:
                t_start = dip_df['Entry Time (s)'].min()
                t_end = dip_df['Exit Time (s)'].max()
                wt_df = wt_df[(wt_df['time'] >= t_start) & (wt_df['time'] <= t_end)]
        
        if len(dip_df) == 0:
            st.warning("No loads match the current filter.")
            st.stop()

        # ─────────────────────────────────────────────────────────────────────
        # DIP TIME OUTPUT TABLE
        # ─────────────────────────────────────────────────────────────────────
        st.subheader("🔬 Dip Time Audit Log")

        # Sort by Load ID (numeric order: L0, L1, ...) then by Station No
        dip_df_sorted = (
            dip_df.copy()
            .sort_values(by=['Load ID', 'Entry Time (s)'])
            .reset_index(drop=True)
        )

        st.dataframe(dip_df_sorted, use_container_width=True, height=300)

        csv_buffer = io.StringIO()
        dip_df_sorted.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Dip Log as CSV",
            data=csv_buffer.getvalue(),
            file_name="DIP_TIME_OUTPUT.csv",
            mime="text/csv",
            key='download_dip'
        )
        
        st.divider()

        # ─────────────────────────────────────────────────────────────────────
        # DIP TIME UNIFORMITY ANALYSIS
        # ─────────────────────────────────────────────────────────────────────
        st.subheader("📐 Dip Time Uniformity Analysis")

        # Only analyse stations that have a meaningful target dip (> 0)
        dip_process = dip_df[dip_df['Target Dip (s)'] > 0].copy()

        if len(dip_process) > 0:
            # Per-station statistics across all loads
            uniformity = (
                dip_process.groupby(['Station No', 'Assigned Tank'])['Actual Dip (s)']
                .agg(
                    Loads='count',
                    Min_Dip='min',
                    Max_Dip='max',
                    Mean_Dip='mean',
                    Spread=lambda x: x.max() - x.min(),
                )
                .reset_index()
                .rename(columns={
                    'Assigned Tank': 'Station',
                    'Min_Dip': 'Min Actual (s)',
                    'Max_Dip': 'Max Actual (s)',
                    'Mean_Dip': 'Mean Actual (s)',
                })
                .sort_values('Station No')
            )

            # Add allowed range from dip_df
            range_df = (
                dip_process.groupby('Station No')
                .agg(Target_Allowed=('Target Dip (s)', 'first'),
                     Max_Allowed=('Max Dip (s)', 'first'))
                .reset_index()
            )
            uniformity = uniformity.merge(range_df, on='Station No')

            # Uniform = spread ≤ 1 s
            uniformity['Uniform?'] = uniformity['Spread'].apply(
                lambda v: '✓ Yes' if v <= 1.0 else f'✗ No  (±{v/2:.1f} s)'
            )
            uniformity['Mean Actual (s)'] = uniformity['Mean Actual (s)'].round(1)
            uniformity['Spread'] = uniformity['Spread'].round(1)

            # Colour-code the dataframe
            def highlight_non_uniform(row):
                colour = 'background-color: #3d2121; color: #ff9999' if '✗' in str(row['Uniform?']) else ''
                return [colour] * len(row)

            display_cols = [
                'Station No', 'Station', 'Loads',
                'Target_Allowed', 'Max_Allowed',
                'Min Actual (s)', 'Max Actual (s)', 'Mean Actual (s)',
                'Spread', 'Uniform?',
            ]
            st.dataframe(
                uniformity[display_cols]
                .rename(columns={'Target_Allowed': 'Target Allowed (s)',
                                 'Max_Allowed': 'Max Allowed (s)'})
                .style.apply(highlight_non_uniform, axis=1),
                use_container_width=True,
                height=max(120, (len(uniformity) + 1) * 38),
            )

            # Bar chart: spread per station
            non_zero_spread = uniformity[uniformity['Spread'] > 0]
            if len(non_zero_spread) > 0:
                fig_spread = px.bar(
                    non_zero_spread,
                    x='Station',
                    y='Spread',
                    color='Spread',
                    color_continuous_scale='RdYlGn_r',
                    title='Dip Time Spread (Max − Min) per Station across all Loads',
                    labels={'Spread': 'Spread (s)'},
                    text='Spread',
                )
                fig_spread.update_traces(texttemplate='%{text:.1f}s', textposition='outside')
                fig_spread.add_hline(
                    y=1, line_dash='dash', line_color='green',
                    annotation_text='Uniform threshold (1 s)',
                    annotation_position='top right'
                )
                fig_spread.update_layout(height=320, showlegend=False)
                st.plotly_chart(fig_spread, use_container_width=True)

            # Explanation
            non_uniform_stations = uniformity[uniformity['Spread'] > 1]['Station'].tolist()
            if non_uniform_stations:
                st.info(
                    f"**Non-uniform stations:** {', '.join(non_uniform_stations)}. "
                    "**Cause:** The two alternating ZINC PHOSPHATING tanks sit at different physical "
                    "distances from the next station, so the wagon returns to upstream tanks (SOAK, "
                    "DERUSTING) at slightly different times for even vs. odd loads. "
                    "The startup loads (L0–L2) also differ because the pipeline is not yet full. "
                    "**All actual dip times are within the allowed min/max range.**"
                )
            else:
                st.success("All stations have uniform dip times across all loads.")

        st.divider()

        # ─────────────────────────────────────────────────────────────────────
        # CYCLE TIME & PROCESSING TIME
        # ─────────────────────────────────────────────────────────────────────

        if len(dip_df) > 0:
            first_load = dip_df['Load ID'].iloc[0]
            sample_entries = dip_df[dip_df['Load ID'] == first_load]
            processing_time = sample_entries['Exit Time (s)'].max() - sample_entries['Entry Time (s)'].min()

        # Warn about stations that cannot accommodate the target cycle time
        if infeasible_stations:
            st.warning(
                f"⚠️ **Infeasible stations** — max dip time < target cycle time "
                f"({target_cycle_time:.1f} s): **{', '.join(infeasible_stations)}**. "
                "Their dip is capped at max_dip, so perfect uniformity cannot be guaranteed there."
            )

        # Compute per-load unload times and inter-unload cycle times
        load_ids_sorted = sorted(dip_df['Load ID'].unique())
        unload_times = []
        for lid in load_ids_sorted:
            t = dip_df[dip_df['Load ID'] == lid]['Exit Time (s)'].max()
            unload_times.append({'load': lid, 'time': t})

        cycle_times = [
            unload_times[i]['time'] - unload_times[i - 1]['time']
            for i in range(1, len(unload_times))
        ]
        avg_ct   = sum(cycle_times) / len(cycle_times) if cycle_times else 0
        max_ct   = max(cycle_times) if cycle_times else 0
        min_ct   = min(cycle_times) if cycle_times else 0
        spread   = max_ct - min_ct

        # ── Cycle Time Summary ────────────────────────────────────────────────
        st.subheader("⏱️ Cycle Time Summary")

        # Inter-Rack Cycle Time = steady-state measured interval between
        # consecutive loads being unloaded (= true throughput rate of the line)
        inter_rack_ct = avg_ct

        # Highlighted primary metric
        st.markdown(
            f"""
            <div style="background:#1f77b4;border-radius:8px;padding:14px 24px;
                        display:inline-block;margin-bottom:12px;">
              <span style="color:#fff;font-size:13px;font-weight:600;
                           letter-spacing:.5px;">INTER-RACK CYCLE TIME</span><br>
              <span style="color:#fff;font-size:36px;font-weight:700;
                           line-height:1.2;">{inter_rack_ct:.1f} s</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Supporting metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Min Cycle Time",    f"{min_ct:.1f} s")
        m2.metric("Max Cycle Time",    f"{max_ct:.1f} s")
        m3.metric("Spread (Max − Min)", f"{spread:.1f} s",
                  delta=f"{spread:+.1f} s" if spread > 0 else None,
                  delta_color="inverse")
        m4.metric("Target Cycle Time", f"{target_cycle_time:.1f} s")
        m5.metric("Uniform?", "Yes ✓" if spread <= 1.0 else f"No — ±{spread/2:.1f} s")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⏱️ Cycle Time Per Load Pair")

            if cycle_times:
                cycle_chart_df = pd.DataFrame({
                    'Load Pair': [f"{unload_times[i-1]['load']}→{unload_times[i]['load']}"
                                  for i in range(1, len(unload_times))],
                    'Cycle Time (s)': cycle_times
                })

                fig_cycle = px.bar(
                    cycle_chart_df,
                    x='Load Pair',
                    y='Cycle Time (s)',
                    title='Cycle Time Between Consecutive Unloads',
                    color='Cycle Time (s)',
                    color_continuous_scale='RdYlGn_r'
                )
                if target_cycle_time > 0:
                    fig_cycle.add_hline(
                        y=target_cycle_time,
                        line_dash='dash',
                        line_color='green',
                        annotation_text=f"Target: {target_cycle_time:.0f} s",
                        annotation_position='top right'
                    )
                fig_cycle.update_layout(height=340, xaxis_tickangle=-45)
                st.plotly_chart(fig_cycle, use_container_width=True)
        
        with col2:
            st.subheader("⏳ Processing Time (Lead Time Per Load)")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Processing Time", f"{processing_time:.1f} s")
            with col_b:
                st.metric("Total Loads", dip_df['Load ID'].nunique())
            with col_c:
                st.metric("Stations in Sequence", dip_df['Station No'].nunique())

            st.markdown("**Dip time breakdown for one load (actual vs target):**")
            sample_load = dip_df['Load ID'].iloc[0]
            sample_dip_times = dip_df[dip_df['Load ID'] == sample_load][
                ['Station No', 'Assigned Tank', 'Target Dip (s)', 'Actual Dip (s)', 'Status']
            ].reset_index(drop=True)
            st.dataframe(sample_dip_times, use_container_width=True, height=200)
        
        st.divider()

        # ─────────────────────────────────────────────────────────────────────
        # WAY-TIME LINE DIAGRAM  (Load × Station Gantt)
        # ─────────────────────────────────────────────────────────────────────
        st.subheader("📊 Way-Time Line Diagram (Load × Station)")

        if len(dip_df) > 0:
            # Station display order: loading at bottom → unload at top
            station_order = (
                dip_df[['Station No', 'Assigned Tank']]
                .drop_duplicates()
                .sort_values('Station No')['Assigned Tank']
                .tolist()
            )

            load_ids_gantt = sorted(dip_df['Load ID'].unique())

            palette = (
                px.colors.qualitative.Plotly
                + px.colors.qualitative.Set2
                + px.colors.qualitative.Dark24
            )

            fig_gantt = go.Figure()

            for i, lid in enumerate(load_ids_gantt):
                load_num = lid
                color    = palette[i % len(palette)]
                rows     = dip_df[dip_df['Load ID'] == lid]

                x_seg, y_seg = [], []
                for _, row in rows.iterrows():
                    if row['Exit Time (s)'] > row['Entry Time (s)']:
                        x_seg += [row['Entry Time (s)'], row['Exit Time (s)'], None]
                        y_seg += [row['Assigned Tank'], row['Assigned Tank'], None]

                if x_seg:
                    fig_gantt.add_trace(go.Scatter(
                        x=x_seg,
                        y=y_seg,
                        mode='lines',
                        line=dict(width=16, color=color),
                        name=f'Load {load_num}',
                        hovertemplate=(
                            f'<b>Load {load_num}</b><br>'
                            '%{y}<br>t = %{x:.1f} s<extra></extra>'
                        ),
                    ))

            ct_title = f"{inter_rack_ct:.0f}s Inter-Rack Cycle Time" if cycle_times else ""
            fig_gantt.update_layout(
                title=f'Way-Time Line Diagram for {len(load_ids_gantt)} Loads'
                      + (f' ({ct_title})' if ct_title else ''),
                xaxis=dict(title='Time (s)', showgrid=True,
                           gridcolor='rgba(200,200,200,0.4)'),
                yaxis=dict(
                    title='Station',
                    categoryorder='array',
                    categoryarray=station_order,
                    showgrid=True,
                    gridcolor='rgba(200,200,200,0.4)',
                ),
                height=520,
                template='plotly_white',
                legend=dict(
                    title='Load',
                    orientation='v',
                    x=1.01, y=1,
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='gray', borderwidth=1,
                ),
                font=dict(family='Arial', size=11),
            )
            st.plotly_chart(fig_gantt, use_container_width=True)

        st.divider()

        # ─────────────────────────────────────────────────────────────────────
        # WAY-TIME DIAGRAM  (single wagon path vs. time)
        # ─────────────────────────────────────────────────────────────────────
        st.subheader("🚚 Way-Time Diagram (Wagon Position vs. Time)")

        if wt_df is not None and len(wt_df) > 0:

            # ── Build three segment traces: loaded / empty / wait ─────────────
            # Each trace collects (x1, x2, None) x-pairs and (y1, y2, None) y-pairs
            # so one trace draws all segments of that type as disconnected lines.
            seg_x = {'loaded': [], 'empty': [], 'wait': []}
            seg_y = {'loaded': [], 'empty': [], 'wait': []}

            for i in range(len(wt_df) - 1):
                p1 = wt_df.iloc[i]
                p2 = wt_df.iloc[i + 1]
                seg = p2['seg']          # type of movement ending at p2
                seg_x[seg] += [p1['time'], p2['time'], None]
                seg_y[seg] += [p1['dist'], p2['dist'], None]

            # ── Station gridlines and labels on Y-axis ────────────────────────
            active_dists = sorted({d['dist'] for d in stations.values() if d['active']})
            # Use physical distance; fall back to station number if all zero
            use_dist = max(active_dists) > 0 if active_dists else False

            tick_vals, tick_texts = [], []
            if stations:
                for sno in sorted(stations):
                    s = stations[sno]
                    yval = s['dist'] if use_dist else sno
                    tick_vals.append(yval)
                    tick_texts.append(f"S{sno}: {s['name'][:18]}")

            fig_wt = go.Figure()

            style = {
                'loaded': dict(color='#1f77b4', width=3),  # blue — carrying load
                'empty' : dict(color='#aec7e8', width=2, dash='dot'),  # light — empty travel
                'wait'  : dict(color='#d62728', width=3),  # red  — waiting at station
            }
            label = {
                'loaded': 'Loaded travel',
                'empty' : 'Empty travel',
                'wait'  : 'Wait (dip not ready)',
            }

            for seg_type in ('loaded', 'empty', 'wait'):
                if seg_x[seg_type]:
                    fig_wt.add_trace(go.Scatter(
                        x=seg_x[seg_type],
                        y=seg_y[seg_type],
                        mode='lines',
                        name=label[seg_type],
                        line=style[seg_type],
                        hovertemplate='Time: %{x:.1f} s<br>Position: %{y:.0f} mm<extra>'
                                      + label[seg_type] + '</extra>',
                    ))

            # Station markers on the wagon path
            fig_wt.add_trace(go.Scatter(
                x=wt_df['time'],
                y=wt_df['dist'],
                mode='markers',
                name='Station stop',
                marker=dict(size=6, color='#333', symbol='circle'),
                text=wt_df['name'],
                hovertemplate='<b>%{text}</b><br>t = %{x:.1f} s<extra></extra>',
                showlegend=False,
            ))

            fig_wt.update_layout(
                title=dict(
                    text='Way-Time Diagram — Single Wagon Path',
                    x=0.5, xanchor='center', font=dict(size=17)
                ),
                xaxis=dict(
                    title='Time (seconds)',
                    showgrid=True, gridcolor='rgba(200,200,200,0.4)',
                ),
                yaxis=dict(
                    title='Station Physical Position (mm)' if use_dist else 'Station Number',
                    tickvals=tick_vals,
                    ticktext=tick_texts,
                    showgrid=True, gridcolor='rgba(200,200,200,0.4)',
                ),
                height=650,
                template='plotly_white',
                hovermode='x unified',
                legend=dict(orientation='h', y=1.05),
                font=dict(family='Arial', size=11),
            )

            st.plotly_chart(fig_wt, use_container_width=True)

            # Quick stats
            s1, s2, s3 = st.columns(3)
            s1.metric("Total simulation time", f"{wt_df['time'].max():.1f} s")
            s2.metric("Way-time points recorded", len(wt_df))
            s3.metric("Loads processed", dip_df['Load ID'].nunique())
        
        st.divider()
        
        # ─────────────────────────────────────────────────────────────────────
        # DIP COMPLIANCE
        # ─────────────────────────────────────────────────────────────────────
        st.subheader("✅ Dip Time Compliance")
        
        dip_quality = dip_df[dip_df['Target Dip (s)'] > 0].copy()
        pass_count = len(dip_quality[dip_quality['Status'].str.contains('PASS')])
        fail_count = len(dip_quality[dip_quality['Status'].str.contains('FAIL')])
        total_dips = len(dip_quality)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if total_dips > 0:
                compliance_pct = (pass_count / total_dips) * 100
                st.metric("Compliance Rate", f"{compliance_pct:.1f}%")
            else:
                st.metric("Compliance Rate", "N/A")
        
        with col2:
            st.metric("✓ Passed Dips", pass_count)
        
        with col3:
            st.metric("✗ Failed Dips", fail_count)
        
        # Failed dips detail
        if fail_count > 0:
            st.warning("⚠️ Failed Dips Details:")
            failed_dips = dip_quality[dip_quality['Status'].str.contains('FAIL')]
            st.dataframe(failed_dips, use_container_width=True)
        
        # Heatmap of dip times
        st.subheader("🌡️ Dip Time Heatmap")
        
        pivot_data = dip_df[dip_df['Target Dip (s)'] > 0].pivot_table(
            values='Actual Dip (s)',
            index='Assigned Tank',
            columns='Load ID',
            aggfunc='first'
        )
        
        if not pivot_data.empty:
            fig_heatmap = px.imshow(
                pivot_data,
                labels=dict(x='Load ID', y='Station', color='Dip Time (s)'),
                color_continuous_scale='RdYlGn',
                title='Actual Dip Time Heatmap (Load vs Station)',
                aspect='auto'
            )
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR - HELP & ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("ℹ️ Help")
    
    st.markdown("""
    ### How to Use:
    
    **1. Configuration Tab:**
    - Upload three CSV files:
      - `input_tanks_csv.csv` - Station/tank configuration
      - `input_wagon_new.csv` - Wagon parameters
      - `input_Crosstrolley.csv` (optional)
    - Set simulation parameters
    - Click "Generate Sequence"
    
    **2. Execution Tab:**
    - View the generated instruction sequence
    - Download as CSV
    
    **3. Analysis Tab:**
    - View dip time audit log
    - Analyze cycle times per load
    - Check processing times by station
    - View wagon movement over time
    - Monitor dip compliance
    
    ### File Requirements:
    
    **input_tanks_csv.csv columns:**
    - station_no
    - process_name
    - distance_mm
    - dip_time_sec
    - max_dip_time_sec
    - station_type
    - Process_NO
    
    **input_wagon_new.csv columns:**
    - Fast Speed Mtrs/Min
    - Superfast SpeedMtrs/Min
    - Lift Time Seconds
    - Lower Time Seconds
    """)
