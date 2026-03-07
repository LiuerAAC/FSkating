import json
import os
from A_Final_rotarion_count import SkeletonAnalyzer

# ==========================================
# 0. Global Configuration
# ==========================================
# Categories defined by ISU technical rules
DIFFICULT_VARIANTS = {
    "CF", "CU", "CS", "SF", "SS", "SB", "UF", "US", "UL", "UB", "UL+Side", "UL+Up", "UL+Knife"
}
EDGE_CHANGE_VARIANTS = {"SF+Inside", "SB+Inside", "CF+Inside", "CS+Inside", "CU+Inside", "BC+Inside"}
SPIN_KEYWORDS = {
    "UF", "US", "UB", "UL", "SF", "SS", "SB", "CF", "CS", "CU",
    "BC", "BS", "BU", "WINDMILL", "SF+Inside", "SB+Inside",
    "CF+Inside", "CS+Inside", "CU+Inside", "BC+Inside"
}


# ==========================================
# 1. Spin Name Determination
# ==========================================
def determine_spin_name(segments):
    """Determines the base code (e.g., SSp, CCoSp) based on labels and transitions."""
    if not segments: return "Unknown"
    labels = [s['label'] for s in segments]
    label_set = set(labels)

    upright_set = {"UF", "US", "UB", "BU", "UL+Side", "UL+Up", "UL+Knife"}
    sit_set = {"SF", "SS", "SB", "SF+Inside", "SB+Inside", "BS"}
    camel_set = {"CF", "CS", "CU", "CF+Inside", "CS+Inside", "CU+Inside", "BC", "BC+Inside"}

    has_u = any(l in upright_set for l in label_set)
    has_s = any(l in sit_set for l in label_set)
    has_c = any(l in camel_set for l in label_set)

    # Combined Spin (CoSp) if 2 or more basic positions are present
    if sum([has_u, has_s, has_c]) >= 2:
        base_name = "CoSp"
    elif has_c:
        base_name = "CSp"
    elif has_s:
        base_name = "SSp"
    else:
        base_name = "USp"

    is_jump = (labels[0] == "Jump In")
    is_change = ("CP" in label_set or "JCP" in label_set)
    prefix = ""
    if is_jump: prefix += "F"  # Flying
    if is_change: prefix += "C"  # Change of foot
    return f"{prefix}{base_name}"


# ==========================================
# 2. Level Determination Core Module
# ==========================================
def calculate_spin_level(segments, analyzer, PROGRAM_STATE):
    """
    Logic Logic Version:
    1. Temporarily ignore exclusion rules for features 1,7,8,9,10,12,13.
    2. Prioritize counting F1 (Difficult Variants).
    3. F4 (Difficult Change of Pos) is only considered if F1 quota is full or redundant.
    """
    labels = [s['label'] for s in segments]
    physics_info = []
    last_leg = None

    # Analyze physics for each segment (revolutions and balance)
    for s in segments:
        info = analyzer.analyze_segment(s['begin'], s['end'], s['label'], last_leg=last_leg)
        physics_info.append(info)
        last_leg = info['leg']

    physics_summary = " -> ".join(
        [f"{s['label']}({physics_info[idx]['leg']}/{physics_info[idx]['revs']}r)" for idx, s in enumerate(segments)])

    potential_features = set()
    foot_variants = {"Left": set(), "Right": set()}

    # --- Phase A: Potential Feature Identification ---
    for i, label in enumerate(labels):
        curr_info = physics_info[i]
        curr_leg = curr_info['leg']
        base_v = label.split('+')[0]

        # Variant recognition
        if label in DIFFICULT_VARIANTS or label in EDGE_CHANGE_VARIANTS or "+Inside" in label:
            if curr_info['revs'] >= 8.0: potential_features.add("F10")  # 8 revs in one position
            foot_variants[curr_leg].add(base_v)
            if "+Inside" in label or label in EDGE_CHANGE_VARIANTS: potential_features.add("F7") # Change of edge

        if len(foot_variants[curr_leg]) >= 2: potential_features.add("F4_POTENTIAL")

    # Jump / Entrance (F11, F2, F3) identification
    if labels[0] == "Jump In" and len(labels) > 1:
        if any(kw in labels[1] for kw in DIFFICULT_VARIANTS) and physics_info[1]['revs'] >= 2.0:
            potential_features.add("F11") # Flying entrance
    if "JCP" in labels:
        for i, l in enumerate(labels):
            if l == "JCP":
                p_idx = next((j for j in range(i - 1, -1, -1) if labels[j] in SPIN_KEYWORDS), -1)
                n_idx = next((j for j in range(i + 1, len(labels)) if labels[j] in SPIN_KEYWORDS), -1)
                if p_idx != -1 and n_idx != -1:
                    if physics_info[p_idx]['leg'] != physics_info[n_idx]['leg']:
                        potential_features.add("F2") # Change of foot via jump
                    else:
                        potential_features.add("F3") # Jump within same foot

    # --- Phase B: Feature Allocation (Prioritizing F1) ---
    final_features = set()
    foot_counts = {"Left": 0, "Right": 0}
    spin_f1_count = 0

    # 1. First Priority: Use F1 difficult variants available in this spin
    for leg in ["Left", "Right"]:
        for v_type in sorted(list(foot_variants[leg])):
            # If variant hasn't been used in the whole program, spin quota < 2, and foot quota < 2
            if v_type not in PROGRAM_STATE["USED_VARIANTS"] and spin_f1_count < 2 and foot_counts[leg] < 2:
                final_features.add(f"F1_{v_type}")
                PROGRAM_STATE["USED_VARIANTS"].add(v_type)
                spin_f1_count += 1
                foot_counts[leg] += 1

    # 2. Second Priority: High-priority non-variant features (F10, F11, F2, F7, etc.)
    priority_pool = ["F10", "F11", "F2", "F3", "F7", "F5", "F6"]
    for f in priority_pool:
        if f in potential_features and f not in PROGRAM_STATE["ONCE_FEATURES"]:
            # Assign to the leg that still has quota
            assigned = False
            for leg in ["Left", "Right"]:
                if foot_counts[leg] < 2:
                    final_features.add(f)
                    PROGRAM_STATE["ONCE_FEATURES"].add(f)
                    foot_counts[leg] += 1
                    assigned = True
                    break
            if not assigned: pass  # No available quota on either foot

    # 3. Third Priority: F4 as fallback
    # Only if: (total features < 4) AND (F4 potential exists) AND (F4 not used yet in program)
    if len(final_features) < 4 and "F4_POTENTIAL" in potential_features:
        if "F4" not in PROGRAM_STATE["ONCE_FEATURES"]:
            for leg in ["Left", "Right"]:
                if foot_counts[leg] < 2:
                    final_features.add("F4")
                    PROGRAM_STATE["ONCE_FEATURES"].add("F4")
                    break

    # --- Phase C: Final Settlement ---
    total_count = len(final_features)
    # Rules specify certain features are required for Level 4
    l4_cores = {"F4", "F6", "F7", "F8", "F9", "F11"}
    has_l4_core = any(c in final_features for c in l4_cores)

    level = 4 if total_count >= 4 and has_l4_core else min(total_count, 3)
    return (level if level > 0 else "B"), final_features, physics_summary

# ==========================================
# 3. Main Function
# ==========================================
def main():
    # Path Configuration
    # JSONL_PATH = 
    # SKELETON_PATH = 
  
    # Initialize program-wide state tracking (to handle unique variants and once-per-program rules)
    PROGRAM_STATE = {
        "F5_F6_DONE": False,  # Difficult entrance/exit lock
        "USED_VARIANTS": set(),  # Difficult variant categories already counted (F1)
        "ONCE_FEATURES": set()  # Global one-time features used (F4, F7, F8, etc.)
    }

    # Initialize Physics Analyzer
    try:
        analyzer = SkeletonAnalyzer(SKELETON_PATH)
    except Exception as e:
        print(f"Failed to initialize SkeletonAnalyzer: {e}")
        return

    print("=" * 85)
    print(f"{'BATCH SPIN TECHNICAL ANALYSIS (ISU 2024-2026)':^85}")
    print("=" * 85)

    if not os.path.exists(JSONL_PATH):
        print(f"JSONL file not found: {JSONL_PATH}")
        return

    # Process spins in batch
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())
            segments = data.get("segments", [])
            spin_id = data.get("spin_id", i + 1)

            # 1. Determine Spin Name (FSSp, CCoSp, etc.)
            base_name = determine_spin_name(segments)

            # 2. Determine Level and Features
            level, feats, summary = calculate_spin_level(segments, analyzer, PROGRAM_STATE)

            # 3. Print analysis results
            full_code = f"{base_name}{level}"
            print(f"Spin {i + 1} | ID: {spin_id:<4} | Final Result: {full_code}")
            print(f"   ▶ Action Sequence: {summary}")
            print(f"   ▶ Counted Features: {sorted(list(feats))}")
            print("-" * 85)

    # 4. Program Summary
    print(f"Program Statistics:")
    print(f"   - Difficult variants used: {sorted(list(PROGRAM_STATE['USED_VARIANTS']))}")
    print(f"   - Global level features triggered: {sorted(list(PROGRAM_STATE['ONCE_FEATURES']))}")
    print("=" * 85)


if __name__ == "__main__":
    main()
