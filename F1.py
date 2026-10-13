import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import warnings, os
warnings.filterwarnings('ignore')

# ── 1. Load Data ───────────────────────────────────────────────────────────────
folder_path = r"C:\Users\acqua\OneDrive\Documents\Attachments\Desktop\codes\summer projects\F1"

results        = pd.read_csv(os.path.join(folder_path, "f1_race_results_2000_2026.csv"))
qualifying     = pd.read_csv(os.path.join(folder_path, "f1_qualifying_2000_2026.csv"))
races          = pd.read_csv(os.path.join(folder_path, "f1_races_2000_2026.csv"))

print("Files loaded.")

# ── 2. Feature Engineering ─────────────────────────────────────────────────────
results['position'] = pd.to_numeric(results['position'], errors='coerce')
results = results.dropna(subset=['position'])
results['win']    = (results['position'] == 1).astype(int)
results['podium'] = (results['position'] <= 3).astype(int)

qual = qualifying[['season','round','driver_id','position']].rename(columns={'position':'quali_pos'})
df = results.merge(qual, on=['season','round','driver_id'], how='left')
df['quali_pos'] = df['quali_pos'].fillna(df['grid'])
df = df.sort_values(['driver_id','season','round'])

# Rolling driver form (last 5 races)
for col, feat in [('position','drv_avg_finish'),('win','drv_win_rate'),
                  ('podium','drv_podium_rate'),('points','drv_avg_pts')]:
    df[feat] = df.groupby('driver_id')[col].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean())

# Rolling constructor form (last 5 races)
con_agg = (df.groupby(['constructor_id','season','round'])[['points','win']]
             .sum().reset_index()
             .sort_values(['constructor_id','season','round']))
for col, feat in [('points','con_avg_pts'),('win','con_win_rate')]:
    con_agg[feat] = con_agg.groupby('constructor_id')[col].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean())
df = df.merge(con_agg[['constructor_id','season','round','con_avg_pts','con_win_rate']],
              on=['constructor_id','season','round'], how='left')

# Circuit encoding
le_circuit = LabelEncoder()
df['circuit_enc'] = le_circuit.fit_transform(df['race_name'])

FEATURES = ['grid','quali_pos','round','circuit_enc',
            'drv_avg_finish','drv_win_rate','drv_podium_rate','drv_avg_pts',
            'con_avg_pts','con_win_rate']

# ── 3. Train Model ────────────────────────────────────────────────────────────
train = df[df['season'] < 2026].dropna(subset=FEATURES+['win'])
model = GradientBoostingClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
model.fit(train[FEATURES], train['win'])
print(f"Model trained on {len(train):,} entries (2000–2025).")

# ── 4. Current 2026 Standings ─────────────────────────────────────────────────
r26 = results[results['season'] == 2026]
driver_standings = (r26.groupby(['driver_id','driver_name','constructor'])['points']
                       .sum().reset_index()
                       .sort_values('points', ascending=False))
constructor_standings = (r26.groupby('constructor')['points']
                            .sum().reset_index()
                            .sort_values('points', ascending=False))

print("\n=== 2026 DRIVER STANDINGS (Rounds 1–3) ===")
print(driver_standings[['driver_name','constructor','points']].to_string(index=False))

print("\n=== 2026 CONSTRUCTOR STANDINGS ===")
print(constructor_standings.to_string(index=False))

# ── 5. Monaco GP Prediction (Round 6) ────────────────────────────────────────
# NOTE: Update grid positions below once qualifying results are available
# Monaco qualifying is critical — pole position is everything at this circuit!

monaco_enc = le_circuit.transform(['Monaco Grand Prix'])[0]
latest_stats = (df[df['season']==2026].sort_values('round')
                  .groupby('driver_id').last().reset_index())
stats_map = latest_stats.set_index('driver_id').to_dict('index')

# ── UPDATE GRID POSITIONS AFTER QUALIFYING ────────────────────────────────────
DRIVERS_2026 = [
    # (driver_id,               display_name,          constructor,         est_grid)
    # Change est_grid to actual qualifying position once known
    ('andrea_kimi_antonelli',  'Kimi Antonelli',       'Mercedes',           1),
    ('george_russell',         'George Russell',       'Mercedes',           2),
    ('charles_leclerc',        'Charles Leclerc',      'Ferrari',            3),
    ('lewis_hamilton',         'Lewis Hamilton',       'Ferrari',            4),
    ('lando_norris',           'Lando Norris',         'McLaren',            5),
    ('oscar_piastri',          'Oscar Piastri',        'McLaren',            6),
    ('max_verstappen',         'Max Verstappen',       'Red Bull',           7),
    ('isack_hadjar',           'Isack Hadjar',         'Red Bull',           8),
    ('pierre_gasly',           'Pierre Gasly',         'Alpine F1 Team',     9),
    ('oliver_bearman',         'Oliver Bearman',       'Haas F1 Team',      10),
    ('liam_lawson',            'Liam Lawson',          'RB F1 Team',        11),
    ('franco_colapinto',       'Franco Colapinto',     'Alpine F1 Team',    12),
    ('carlos_sainz',           'Carlos Sainz',         'Williams',          13),
    ('esteban_ocon',           'Esteban Ocon',         'Haas F1 Team',      14),
    ('arvid_lindblad',         'Arvid Lindblad',       'RB F1 Team',        15),
    ('nico_hulkenberg',        'Nico Hulkenberg',      'Audi',              16),
    ('gabriel_bortoleto',      'Gabriel Bortoleto',    'Audi',              17),
    ('alexander_albon',        'Alexander Albon',      'Williams',          18),
    ('fernando_alonso',        'Fernando Alonso',      'Aston Martin',      19),
    ('lance_stroll',           'Lance Stroll',         'Aston Martin',      20),
    ('sergio_perez',           'Sergio Perez',         'Cadillac F1 Team',  21),
    ('valtteri_bottas',        'Valtteri Bottas',      'Cadillac F1 Team',  22),
]

rows = []
for did, dname, con, grid in DRIVERS_2026:
    s = stats_map.get(did, {})
    rows.append({
        'driver_id': did, 'driver_name': dname, 'constructor': con,
        'grid': grid, 'quali_pos': grid, 'round': 6,
        'circuit_enc':      monaco_enc,
        'drv_avg_finish':   s.get('drv_avg_finish',   12),
        'drv_win_rate':     s.get('drv_win_rate',       0),
        'drv_podium_rate':  s.get('drv_podium_rate',    0),
        'drv_avg_pts':      s.get('drv_avg_pts',        0),
        'con_avg_pts':      s.get('con_avg_pts',        0),
        'con_win_rate':     s.get('con_win_rate',       0),
    })

monaco_df = pd.DataFrame(rows).fillna(0)
monaco_df['win_prob'] = model.predict_proba(monaco_df[FEATURES])[:,1]
monaco_df['win_prob'] = monaco_df['win_prob'] / monaco_df['win_prob'].sum()
monaco_df = monaco_df.sort_values('win_prob', ascending=False).reset_index(drop=True)

print("\n=== MONACO GP — WIN PROBABILITY (Round 6) ===")
print(f"{'Driver':<26} {'Constructor':<20} {'Grid':>4}  {'Win Prob':>8}")
print("-" * 65)
for _, r in monaco_df.iterrows():
    print(f"  {r['driver_name']:<24} {r['constructor']:<20} {int(r['grid']):>4}  {r['win_prob']:>8.1%}")

# ── 6. Championship Simulation (10,000 runs) ──────────────────────────────────
CURRENT_PTS = {
    'andrea_kimi_antonelli': 68, 'george_russell': 55, 'charles_leclerc': 42,
    'lewis_hamilton': 35,        'lando_norris': 20,   'oscar_piastri': 18,
    'oliver_bearman': 16,        'pierre_gasly': 15,   'max_verstappen': 12,
    'liam_lawson': 8,            'isack_hadjar': 4,    'arvid_lindblad': 4,
    'gabriel_bortoleto': 2,      'carlos_sainz': 2,    'franco_colapinto': 1,
    'esteban_ocon': 1,           'alexander_albon': 0, 'fernando_alonso': 0,
    'lance_stroll': 0,           'nico_hulkenberg': 0, 'sergio_perez': 0,
    'valtteri_bottas': 0,
}

REMAINING_ROUNDS = list(range(6, 23))   # Rounds 6–22 = 17 races including Monaco
PTS_MAP = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}

driver_ids   = [r[0] for r in DRIVERS_2026]
driver_names = {r[0]: r[1] for r in DRIVERS_2026}
driver_cons  = {r[0]: r[2] for r in DRIVERS_2026}
base_probs   = monaco_df.set_index('driver_id')['win_prob'].to_dict()

N_SIM = 10_000
drv_wins = {did: 0 for did in driver_ids}
con_wins = {}
np.random.seed(42)

print(f"\nRunning {N_SIM:,} championship simulations...")
for _ in range(N_SIM):
    sim_pts = dict(CURRENT_PTS)
    for rnd in REMAINING_ROUNDS:
        probs = np.array([
            max(base_probs.get(did, 0.001) + np.random.normal(0, 0.005), 0.0001)
            for did in driver_ids
        ])
        probs /= probs.sum()
        finish = np.random.choice(driver_ids, size=len(driver_ids), replace=False, p=probs)
        for pos, did in enumerate(finish[:10], 1):
            sim_pts[did] = sim_pts.get(did, 0) + PTS_MAP[pos]

    winner = max(sim_pts, key=lambda d: sim_pts[d])
    drv_wins[winner] = drv_wins.get(winner, 0) + 1

    con_sim = {}
    for did, pts in sim_pts.items():
        c = driver_cons.get(did, 'Unknown')
        con_sim[c] = con_sim.get(c, 0) + pts
    cw = max(con_sim, key=lambda c: con_sim[c])
    con_wins[cw] = con_wins.get(cw, 0) + 1

print(f"\n=== 2026 DRIVER CHAMPIONSHIP PROBABILITY ({N_SIM:,} simulations) ===")
print(f"{'Driver':<26} {'Constructor':<18} {'Current Pts':>11}  {'Probability':>11}")
print("-" * 72)
champ_rows = [
    {'driver': driver_names.get(k, k), 'constructor': driver_cons.get(k,''),
     'current_pts': CURRENT_PTS.get(k, 0), 'prob': v / N_SIM}
    for k, v in drv_wins.items() if v > 0
]
for r in sorted(champ_rows, key=lambda x: -x['prob']):
    print(f"  {r['driver']:<24} {r['constructor']:<18} {r['current_pts']:>11.0f}  {r['prob']:>10.2%}")

print(f"\n=== 2026 CONSTRUCTOR CHAMPIONSHIP PROBABILITY ({N_SIM:,} simulations) ===")
print(f"{'Constructor':<22} {'Probability':>11}")
print("-" * 35)
con_rows = [{'constructor': k, 'prob': v / N_SIM} for k, v in con_wins.items() if v > 0]
for r in sorted(con_rows, key=lambda x: -x['prob']):
    print(f"  {r['constructor']:<20} {r['prob']:>10.2%}")

print("\nDone. Tip: update grid positions in DRIVERS_2026 after Monaco qualifying. nn")
