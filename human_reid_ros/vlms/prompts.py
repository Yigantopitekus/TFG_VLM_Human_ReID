"""Prompts de re-identificación (solo ropa y accesorios, sin rasgos faciales)."""

REFERENCES_PROMPT = (
    "Please analyze the clothing, apparel, and accessories worn by the pedestrian "
    "in these images. Provide a highly detailed list of the garments, colors, and footwear. "
    "Do not describe facial features, focus strictly on the clothing inventory."
)

QUERY_PROMPT = (
    "Analyze the garments and clothing worn by the pedestrian in this new image. "
    "Does this clothing inventory match the exact apparel description you just provided? "
    "Do not attempt to identify the individual, strictly compare the clothing items and colors. "
    "Start your answer with 'MATCH' if the clothing is the same, or 'NO MATCH' if it is different, "
    "followed by a brief visual justification."
)
