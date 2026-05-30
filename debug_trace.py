"""Debug trace: run the scheduler and record every move between t=2800 and t=3600."""
import math, pandas as pd
from collections import Counter

def safe_float(val, default=0.0):
    try:
        x = float(val)
        return default if math.isnan(x) else x
    except (TypeError, ValueError):
        return default

tanks_df = pd.read_csv('input_tanks_csv.csv')
wagon_df = pd.read_csv('input_wagon_new.csv')
tanks_df.columns = [c.strip() for c in tanks_df.columns]
wagon_df.columns = [c.strip() for c in wagon_df.columns]

w = wagon_df.iloc[0]
FAST_SPD = safe_float(w.get('Fast Speed Mtrs/Min', 0)) * 1000 / 60
SUPER_SPD = safe_float(w.get('Superfast SpeedMtrs/Min', 0)) * 1000 / 60
SLOW_SPD = safe_float(w.get('Slow Speed Mtrs/Min', 0)) * 1000 / 60
LIFT_T = safe_float(w.get('Lift Time Seconds', 0))
LOWER_T = safe_float(w.get('Lower Time Seconds', 0))
SLOW_ZONE_MM = 300.0

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
        'name': str(row.get('process_name', f'Station {sno}')).strip(),
        'dist': safe_float(row.get('distance_mm', 0)),
        'dip': safe_float(row.get('dip_time_sec', 0)),
        'max_dip': mx if mx > 0 else float('inf'),
        'stype': str(row.get('station_type', '')).strip(),
        'active': active,
        'pno': int(float(pno_raw)) if active else None,
        'can_rest': can_rest_raw in ('yes', 'true', '1'),
    }

active_snos = sorted(s for s, d in stations.items() if d['active'])
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
    if is_load and LOAD_SNO is None: LOAD_SNO = sno
    if is_unload and UNLOAD_SNO is None: UNLOAD_SNO = sno
if LOAD_SNO is None: LOAD_SNO = active_snos[0]
if UNLOAD_SNO is None: UNLOAD_SNO = active_snos[-1]
is_circular = (LOAD_SNO == UNLOAD_SNO)

def travel_t(from_sno, to_sno, loaded=True, lift=False, lower=False):
    dist = abs(stations[to_sno]['dist'] - stations[from_sno]['dist'])
    spd_main = FAST_SPD if loaded else SUPER_SPD
    slow = SLOW_SPD if SLOW_SPD > 0 else spd_main
    if dist <= SLOW_ZONE_MM:
        traverse = dist / slow
    else:
        traverse = (dist - SLOW_ZONE_MM) / spd_main + SLOW_ZONE_MM / slow
    return traverse + (LIFT_T if lift else 0.0) + (LOWER_T if lower else 0.0)

def effective_ready_dip(sno):
    mn = stations[sno]['dip']
    mx = stations[sno]['max_dip']
    target = LOWER_T + mn
    return target if mx == float('inf') else min(target, LOWER_T + mx)

pno_to_alts = {}
for s in alt_tanks:
    pno_to_alts.setdefault(stations[s]['pno'], []).append(s)

zp_toggle = 0

def peek_dest(from_sno):
    idx = active_snos.index(from_sno)
    if from_sno in alt_tanks:
        pno = stations[from_sno]['pno']
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
        pno = stations[nxt]['pno']
        group = sorted([s for s in alt_tanks if stations[s]['pno'] == pno])
        return group[zp_toggle % len(group)]
    return nxt

def consume_dest(from_sno):
    global zp_toggle
    dest = peek_dest(from_sno)
    if dest in alt_tanks:
        zp_toggle += 1
    return dest

def crit_rank_fn(sno):
    c = stations[sno].get('Criticality', 'LOW').upper()
    return {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(c, 1)

n_flightbars = 3
total_loads = 10
n_warmup = n_flightbars
total_sim_loads = n_warmup + total_loads + n_warmup

def _make_lid(idx):
    if idx < n_warmup:
        return 900 + idx + 1
    if idx < n_warmup + total_loads:
        return idx - n_warmup + 1
    return 800 + (idx - n_warmup - total_loads + 1)

fb_pool = list(range(1, n_flightbars + 1))
fb_assignment = {}
empty_fbs = {}
tank_contents = {}
clock = 0.0
wagon_pos = LOAD_SNO
next_load_idx = 0
unloaded_count = 0

def _park_empty(sno, fb_id, avail_time=0.0):
    empty_fbs.setdefault(sno, []).append({'id': fb_id, 'avail_time': avail_time})

def _try_seed():
    global next_load_idx
    if LOAD_SNO not in tank_contents and fb_pool and next_load_idx < total_sim_loads:
        new_fb = fb_pool.pop(0)
        new_lid = _make_lid(next_load_idx)
        fb_assignment[new_lid] = new_fb
        tank_contents[LOAD_SNO] = {'load_id': new_lid, 'entry_time': clock, 'fb_id': new_fb}
        next_load_idx += 1

_try_seed()

trace = []
DEBUG_START = 2850
DEBUG_END = 3600

for _ in range(200_000):
    pending_fbs = sum(len(v) for v in empty_fbs.values())
    if unloaded_count >= total_sim_loads and pending_fbs == 0:
        break

    best_sno = None
    best_score = (float('inf'), float('inf'), float('inf'), float('inf'))

    for sno in sorted(tank_contents.keys(), reverse=True):
        dest = peek_dest(sno)
        if dest is None:
            continue
        if dest in tank_contents:
            continue
        lower_end_t = tank_contents[sno]['entry_time'] + LOWER_T
        target_lift_t = lower_end_t + stations[sno]['dip']
        travel_to_src = travel_t(wagon_pos, sno, loaded=False, lift=True)
        arrive = clock + travel_to_src
        pickup = max(arrive, target_lift_t)
        end_t = pickup + travel_t(sno, dest, loaded=True, lower=True)
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
                        chain_free_t += c_dip + travel_t(curr_chk, active_snos[nxt_idx], loaded=True, lower=True)
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
            deadline = other_c['entry_time'] + LOWER_T + mx
            arrive_after = end_t + travel_t(dest, other_sno, loaded=False, lift=True)
            if arrive_after > deadline:
                violations += (arrive_after - deadline)
            if chain_free_t > deadline:
                violations += (chain_free_t - deadline)
        mx_own = stations[sno]['max_dip']
        dip_so_far = max(0.0, clock - lower_end_t)
        urgency = dip_so_far / mx_own if mx_own not in (float('inf'), 0) else 0.0
        score = (-crit_rank_fn(sno), -urgency, violations, pickup)
        if score < best_score:
            best_score = score
            best_sno = sno

    best_empty_sno = None
    best_empty_score = (float('inf'), float('inf'))
    for sno, fb_list in empty_fbs.items():
        if not fb_list:
            continue
        fb_info = fb_list[0]
        ready_at_fb = fb_info['avail_time']
        arrive_est = clock + travel_t(wagon_pos, sno, loaded=False, lift=True)
        arrive = max(arrive_est, ready_at_fb)
        end_t = arrive + travel_t(sno, LOAD_SNO, loaded=True, lower=True)
        violations = 0.0
        for other_sno, other_c in tank_contents.items():
            mx = stations[other_sno]['max_dip']
            if mx == float('inf'):
                continue
            deadline = other_c['entry_time'] + LOWER_T + mx
            arrive_after = end_t + travel_t(LOAD_SNO, other_sno, loaded=False, lift=True)
            if arrive_after > deadline:
                violations += (arrive_after - deadline)
        score = (violations, arrive)
        if score < best_empty_score:
            best_empty_score = score
            best_empty_sno = sno

    if best_sno is None and best_empty_sno is None:
        if tank_contents:
            soonest = min(v['entry_time'] + effective_ready_dip(k) for k, v in tank_contents.items())
        else:
            soonest = float('inf')
        soonest_empty = min((fl[0]['avail_time'] for fl in empty_fbs.values() if fl), default=float('inf'))
        soonest = min(soonest, soonest_empty)
        clock = max(clock, soonest) + 0.1 if soonest != float('inf') else clock + 0.1
        continue

    loading_stalled = (
        LOAD_SNO not in tank_contents and not fb_pool
        and pending_fbs > 0 and next_load_idx < total_loads
    )
    empty_viol = best_empty_score[0]
    loaded_viol = best_score[2] if best_sno is not None else float('inf')
    do_empty = (
        (best_sno is None and best_empty_sno is not None)
        or (loading_stalled and best_empty_sno is not None and empty_viol <= loaded_viol)
    )

    if DEBUG_START <= clock <= DEBUG_END:
        tc = {k: '%s(Le=%.0f,tgt=%.0f,max=%.0f)' % (
            stations[k]['name'][:4],
            tank_contents[k]['entry_time'] + LOWER_T,
            tank_contents[k]['entry_time'] + LOWER_T + stations[k]['dip'],
            tank_contents[k]['entry_time'] + LOWER_T + stations[k]['max_dip'] if stations[k]['max_dip'] != float('inf') else 99999
        ) for k in tank_contents}
        ef = {k: '%s(avail=%.0f)' % (stations[k]['name'][:4], empty_fbs[k][0]['avail_time']) for k in empty_fbs if empty_fbs[k]}
        chosen = ('EMPTY:%s' % stations[best_empty_sno]['name'] if do_empty else
                  ('LOADED:%s' % stations[best_sno]['name'] if best_sno else 'NONE'))
        trace.append('t=%.1f pos=%s(%s) do=%s tanks=%s empty=%s' % (
            clock, wagon_pos, stations[wagon_pos]['name'][:6], chosen, tc, ef))

    if do_empty:
        e_sno = best_empty_sno
        fb_info = empty_fbs[e_sno].pop(0)
        fb_id = fb_info['id']
        if not empty_fbs[e_sno]:
            del empty_fbs[e_sno]
        ready_at_fb = fb_info['avail_time']
        travel_to_fb = travel_t(wagon_pos, e_sno, loaded=False, lift=True)
        arrive_empty = clock + travel_to_fb
        wait_for_fb = max(0.0, ready_at_fb - arrive_empty)
        if wait_for_fb > 0.1:
            clock += wait_for_fb
        clock += travel_to_fb
        wagon_pos = e_sno
        park_sno = None
        if park_sno is not None:
            tt = travel_t(e_sno, park_sno, loaded=True, lower=True)
            clock += tt
            wagon_pos = park_sno
            _park_empty(park_sno, fb_id, clock)
        else:
            tt = travel_t(e_sno, LOAD_SNO, loaded=True, lower=True)
            clock += tt
            wagon_pos = LOAD_SNO
            fb_pool.append(fb_id)
            _try_seed()
    else:
        if best_sno is None:
            soonest = min(v['entry_time'] + effective_ready_dip(k) for k, v in tank_contents.items())
            clock = max(clock, soonest) + 0.1
            continue

        content = tank_contents[best_sno]
        load_id = content['load_id']
        fb_id = content['fb_id']
        lower_end_t = content['entry_time'] + LOWER_T
        target_lift_t = lower_end_t + stations[best_sno]['dip']
        travel_to_src = travel_t(wagon_pos, best_sno, loaded=False, lift=True)
        required_dep = target_lift_t - travel_to_src
        wait_secs = max(0.0, required_dep - clock)
        if wait_secs > 0.1:
            clock += wait_secs
        clock += travel_to_src
        pickup_time = clock
        dest = consume_dest(best_sno)
        tt = travel_t(best_sno, dest, loaded=True, lower=True)
        clock += tt
        entry_time = clock
        old_entry = content['entry_time']
        del tank_contents[best_sno]
        wagon_pos = dest

        if dest == UNLOAD_SNO:
            if is_circular:
                dip_req = stations[UNLOAD_SNO]['dip']
                unloaded_time = entry_time + LOWER_T + dip_req
                unloaded_count += 1
                if next_load_idx < total_sim_loads:
                    new_lid = _make_lid(next_load_idx)
                    fb_assignment[new_lid] = fb_id
                    tank_contents[LOAD_SNO] = {
                        'load_id': new_lid, 'entry_time': unloaded_time - dip_req, 'fb_id': fb_id
                    }
                    next_load_idx += 1
                else:
                    _park_empty(UNLOAD_SNO, fb_id, unloaded_time)
            else:
                unloaded_count += 1
                _park_empty(UNLOAD_SNO, fb_id, entry_time + LOWER_T)
        else:
            tank_contents[dest] = {'load_id': load_id, 'entry_time': entry_time, 'fb_id': fb_id}

        _try_seed()

for t in trace:
    print(t)
