from tools.search_tools import SearchTool
from config.settings import tavily_cfg

s_tool = SearchTool(cfg=tavily_cfg)


result = s_tool.search(topic='Cricket')

print(result)
print(type(result))
