This text contains the instructions on how to classify the burn injuries in the patient described in the medical note.
The instructions are in english, the texts are in european portuguese. Some explanations are within () such as (in PT:).

# how to classifcy burn depth
- first-degree burn (in PT: queimadura de 1º grau, 1ºG). Always superficial.
- second-degree burn (in PT: queimadura de 2º grau, 2ºG). Can be either:
    - superficial : it has blisters, is painful, is moist.
    - deep (in PT: profunda) : blisters easily unroofed, less painful, dry
- third-degree (in PT: queimadura de 3º grau, 3ºG). full thickness.
- fourth-degree (in PT: queimadura de 4ª grau, 4ºG). extends beyond skin to involve underlying tissues, such as muscle and bone.

# how to classifuy burn size
- described in percentage of body surface area
- sum it up with total body surface area - TBSA (in PT: área total de superfície corporal, ASC), burned body surface area (in PT: ASCQ)
- the percentage % sign is often used
- sometimes the number is preceded by tilde ~ to mean "roughly" (cerca de)

# how to classify burn location
- described as body locations
- the locations are described in the following markdown table with 2 columns. The first column is the burn location. The second column is the SNOMED CT code:

| Burn Location | SNOMED CT Concept (ID) |
|---|---|
| Head and Neck | |
| Scalp | 41695006 |
| Face | 89545001 |
| Ear | 117590005 |
| Eye | 81745001 |
| Nose | 45206002 |
| Mouth | 123851003 |
| Neck | 45048000 |
| Upper Extremities | |
| Shoulder | 16982005 |
| Upper arm | 40983000 |
| Elbow | 127949000 |
| Forearm | 14975008 |
| Wrist | 8205005 |
| Hand | 85562004 |
| Fingers | 7569003 |
| Trunk | |
| Thorax | 51185008 |
| Abdomen | 818983003 |
| Back of trunk | 77568009 |
| Genital structure | 71934003 |
| Perineum | 38864007 |
| Lower Extremities | |
| Hip | 29836001 |
| Thigh | 68367000 |
| Knee | 72696002 |
| Lower leg | 30021000 |
| Ankle | 344001 |
| Foot | 56459004 |
| Toes | 29707007 |

- If the location is stated in the text but is not specific enough, classify it using this list, according to the location:
    - Head and neck: unspecified
    - Upper extremities: unspecified
    - Trunk: unspecified
    - Lower extremities: unspecified

BURN LOCATION CLASSIFICATION
----------------------------
Locations to identify:
- Head and neck
- Upper extremities
- Trunk
- Lower extremities

Values for each location:
- Percentage: integer number (0-100)
- If not specified: return "unspecified"

If the patient has had skin havested from another body part, do not include that part as burned. It is a donor part (in PT: dador, doador, colhido, colheita)

# how to classify circunferencial burns
- burns can only be circumferential in this locations:
    - in a limb 
    - in the torso
- return "yes" if in those locations and specified in the text that it is circumferencial.
- return "no" if in those locations and specified in the text that it is not circumferencial.
- return "unspecifiec" if in those locations and not specified in the text whether it is or not circumferencial.
- if not in those locations, return NULL

CIRCUMFERENTIAL BURNS
--------------------
Applicable locations:
- Limbs (upper/lower extremities)
- Torso (trunk)

Return values:
- "yes": if explicitly mentioned as circumferential
- "no": if explicitly mentioned as not circumferential
- "unknown": if presence cannot be determined
- null: if not applicable location (e.g., head)

Example interpretations:
- "circumferential burn in left arm" -> "yes"
- "non-circumferential trunk burn" -> "no"
- "burn in right leg, extent unclear" -> "unknown"
- "burn in face" -> null

LATERALITY CLASSIFICATION
------------------------
Applicable locations:
- Upper extremities (arms, hands, fingers)
- Lower extremities (legs, feet, toes)

Return values:
- "left": if explicitly mentioned as left side
- "right": if explicitly mentioned as right side
- "bilateral": if both sides are mentioned
- "unspecified": if side is not mentioned
- null: if not applicable location (e.g., trunk, head)

Example interpretations:
- "burn in left arm" -> "left"
- "burns in both legs" -> "bilateral"
- "right hand affected" -> "right"
- "burn in chest" -> null
- "foot burn" -> "unspecified"

# how to classify the mechanism of injury
Use the following list with type and subtype. If multiple types in the same patient, return the most relevant:
- thermal:
    - flame
    - scald
    - contact
- electrical:
    - high voltage
    - low voltage
- chemical:
    - acid
    - alkaline
    - organic solvents
- radiation:
    - sunburn
    - radiation

MECHANISM OF INJURY
------------------
Types and subtypes (multiple possible):
1. Thermal
   - flame
   - scald
   - contact
2. Electrical
   - high voltage
   - low voltage
3. Chemical
   - acid
   - alkaline
   - organic solvents
4. Radiation
   - sunburn
   - radiation

# specific etiologic agent causing the burn
- try to find in the text the specific cause of the flame, eletrical source, and chemical agents
- if not present in the text return "unspecified"

ETIOLOGIC AGENT
--------------
Return specific cause:
- For flame: ignition source
- For electrical: power source
- For chemical: specific agent
Default: "unspecified"

# how to classify inhalation associated problems
- inhalation of:
    - smoke
    - toxic fumes
    - monoxide carbon intoxication(in PT: CO, monoxido de carbono)
    - cyanide (in PT: cianeto)
- heat lesion of the airway:
    - heat lesions of airways (in PT: lesão inalatória)

INHALATION PROBLEMS
------------------
Types:
1. Inhalation of:
   - smoke
   - toxic fumes
   - carbon monoxide (CO, monóxido de carbono)
   - cyanide (cianeto)
2. Airway heat lesions:
   - heat lesions (lesão inalatória)

For each type:
- Return "yes" if present
- Return "no" if explicitly absent
- Return "unspecified" if not mentioned

