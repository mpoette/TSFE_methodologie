"""Literature-friendly display names for pipeline features."""

from __future__ import annotations

import re


FEATURE_NAME_MAPPING: dict[str, tuple[str, str]] = {
    "score_glasgow": ("Glasgow Coma Scale", "GCS"),
    "is_conscious": ("Consciousness", "Consciousness"),
    "heart_rate": ("Heart rate", "HR"),
    "creat": ("Creatinine", "Creat."),
    "is_cvvhf": ("Renal replacement therapy", "RRT"),
    "is_hdi": ("Renal replacement therapy (Intermittent hemodialysis)", "IHD"),
    "pao2": ("Partial pressure of arterial oxygen", "PaO2"),
    "is_ventilated": ("Mechanical ventilation", "MV"),
    "fio2_corr": ("Fraction of inspired oxygen", "FiO2"),
    "age": ("Age", "Age"),
    "temp": ("Temperature", "Temp."),
    "urine_rate": ("Urine output", "Urine"),
    "pas": ("Systolic blood pressure", "SBP"),
    "pam": ("Mean arterial pressure", "MAP"),
    "pad": ("Diastolic blood pressure", "DBP"),
    "bili_tot": ("Total bilirubin", "Bili."),
    "leucocytes": ("White blood cell count", "WBC"),
    "fr": ("Respiratory rate", "RR"),
    "ph": ("Arterial pH", "pH"),
    "sodium": ("Sodium", "Na"),
    "potassium": ("Potassium", "K"),
    "num_plq": ("Platelet count", "PLT"),
    "blood_urea": ("Blood urea nitrogen", "BUN"),
    "nad_dose_poids": ("Norepinephrine dose (mcg/kg/min)", "NE"),
    "dobu_dose_poids": ("Dobutamine dose (mcg/kg/min)", "DOB"),
    "hemoglobine": ("Hemoglobin", "Hb"),
    "tp": ("Prothrombin time", "PT"),
    "spo2": ("Oxygen saturation", "SpO2"),
    "hco3": ("Bicarbonate", "HCO3-"),
    "glyc_cap": ("Capillary blood glucose", "CBG"),
    "hx_respi_chronique": ("Chronic comorbidities (Chronic respiratory disease)", "Chronic Resp."),
    "hx_cirrhose": ("Chronic comorbidities (Cirrhosis)", "Cirrhosis"),
    "hx_cancer": ("Chronic comorbidities (Cancer)", "Cancer"),
    "hx_insuff_cardiaque": ("Chronic comorbidities (Chronic heart failure)", "CHF"),
    "hx_irc": ("Chronic comorbidities (Chronic renal failure)", "CRF"),
    "hx_ttt_immunosuppresseur": ("Treatment on admission (Immunosuppressive therapy)", "IST"),
    "admission_type_Medical": ("Admission type (Medical)", "Medical"),
    "admission_type_Scheduled Surgery": ("Admission type (Scheduled surgery)", "Scheduled Surg."),
    "admission_type_Unknown": ("Admission type (Unknown)", "Unknown"),
    "admission_type_Unscheduled Surgery": ("Admission type (Unscheduled surgery)", "Unscheduled Surg."),
    "icu_ghm_Neurosurgery and neuro-embolisation": ("Primary admission diagnosis (Neurosurgery and neuro-embolisation)", "Neurosurgery"),
    "icu_ghm_Neurology": ("Primary admission diagnosis (Neurology)", "Neurology"),
    "icu_ghm_Others": ("Primary admission diagnosis (Others)", "Other Dx"),
    "icu_ghm_ENT Surgery": ("Primary admission diagnosis (ENT surgery)", "ENT Surg."),
    "icu_ghm_Respiratory Pathology": ("Primary admission diagnosis (Respiratory pathology)", "Respiratory"),
    "icu_ghm_Cardiovascular Surgery": ("Primary admission diagnosis (Cardiovascular surgery)", "CV Surg."),
    "icu_ghm_Cardiology": ("Primary admission diagnosis (Cardiology)", "Cardiology"),
    "icu_ghm_Gastroenterology": ("Primary admission diagnosis (Gastroenterology)", "Gastro."),
    "icu_ghm_Orthopedic Surgery and Amputations": ("Primary admission diagnosis (Orthopedic surgery and amputations)", "Ortho. Surg."),
    "icu_ghm_Metabolic and renal disorders": ("Primary admission diagnosis (Metabolic and renal disorders)", "Metab./Renal"),
    "icu_ghm_Urological Surgery": ("Primary admission diagnosis (Urological surgery)", "Urol. Surg."),
    "icu_ghm_Infectious Disease and Sepsis": ("Primary admission diagnosis (Infectious disease and sepsis)", "Infection/Sepsis"),
    "icu_ghm_Poisoning": ("Primary admission diagnosis (Poisoning)", "Poisoning"),
    "icu_ghm_Polytrauma": ("Primary admission diagnosis (Polytrauma)", "Polytrauma"),
    "icu_ghm_Solid Organ Transplantation": ("Primary admission diagnosis (Solid organ transplantation)", "SOT"),
    "icu_mode_entree_Home or Emergency": ("Patient origin prior to ICU admission (Home or Emergency)", "Home/ED"),
    "icu_mode_entree_Hospital Transfer": ("Patient origin prior to ICU admission (Hospital transfer)", "Hosp. Transfer"),
    "icu_mode_entree_Inter-hospital Transfer": ("Patient origin prior to ICU admission (Inter-hospital transfer)", "Inter-hosp."),
    "icu_mode_entree_Others": ("Patient origin prior to ICU admission (Others)", "Other Origin"),
}

_FEATURE_KEYS_BY_LENGTH = sorted(FEATURE_NAME_MAPPING, key=len, reverse=True)


def short_feature_name(feature_name: object, include_suffix: bool = True) -> str:
    """Return the short literature label for a pipeline feature name.

    Exact feature names use the second element of their mapping tuple. Derived
    TSFEL names retain their descriptor after the mapped source label, while
    unknown names remain unchanged.

    Args:
        feature_name: Internal pipeline feature name.
        include_suffix: Whether to retain a derived-feature suffix.

    Returns:
        Short display label suitable for a figure axis or legend.
    """
    name = str(feature_name)
    exact = FEATURE_NAME_MAPPING.get(name)
    if exact is not None:
        return exact[1]

    for key in _FEATURE_KEYS_BY_LENGTH:
        prefix = f"{key}_"
        if not name.startswith(prefix):
            continue
        short_name = FEATURE_NAME_MAPPING[key][1]
        if not include_suffix:
            return short_name
        suffix = name[len(prefix):]
        readable_suffix = re.sub(r"[_\s]+", " ", suffix).strip()
        return f"{short_name} — {readable_suffix}" if readable_suffix else short_name
    return name
