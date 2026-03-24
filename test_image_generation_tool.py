# from tools.image_generator import ImageGeneratorTool
from tools.image_generator import ImageGeneratorTool
from config.settings import image_gen_cfg

tool_image_obj = ImageGeneratorTool(cfg=image_gen_cfg)

# prompt = (
#     "dynamic full-body shot of Spider-Man mid-swing between skyscrapers, "
#     "classic red and blue suit, web strands catching golden hour sunlight, "
#     "New York City skyline background, motion blur on webs, "
#     "dramatic upward angle, lens flare, photorealistic, cinematic, 8k"
# )
prompt = "medium shot of a modern office workstation, a large language model interface displayed on a transparent screen, a cybersecurity analyst reviewing logs while a subtle silhouette of a hacker watches from a dark doorway, city skyline at dusk through floor‑to‑ceiling windows, cool blue ambient lighting with a hint of red warning alerts, 35mm lens, shallow depth of field, photorealistic, cinematic, 8k"

result = tool_image_obj.generate(prompt=prompt)