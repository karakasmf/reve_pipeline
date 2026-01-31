from typing import Dict, List, Optional, Tuple

def load_labels(participants_txt: str, target_labels: Optional[List[str]] = None) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, str]]:
    """
    participants.txt: whitespace columns: <subject_id> <label_str>
    Returns:
      label_dict: {sid: class_id}
      label_map:  {label_str: class_id}
      id_to_label:{class_id: label_str}

    If target_labels given, class ids follow that order (filtered by what exists).
    """
    mapping_raw: Dict[str, str] = {}
    with open(participants_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            sid, lab = parts[0], parts[1]
            mapping_raw[sid] = lab

    if target_labels is not None:
        target_labels = list(target_labels)
        mapping_raw = {sid: lab for sid, lab in mapping_raw.items() if lab in target_labels}

    if not mapping_raw:
        raise ValueError("No subjects remain after filtering TARGET_LABELS. Check participants.txt and TARGET_LABELS.")

    present = set(mapping_raw.values())
    if target_labels is not None:
        uniq = [lab for lab in target_labels if lab in present]
    else:
        uniq = sorted(present)

    if len(uniq) < 2:
        raise ValueError(f"Need at least 2 classes. Remaining unique labels: {uniq}")

    label_map = {lab: i for i, lab in enumerate(uniq)}
    id_to_label = {i: lab for lab, i in label_map.items()}
    label_dict = {sid: label_map[lab] for sid, lab in mapping_raw.items()}
    return label_dict, label_map, id_to_label
