from duckduckgo_search import DDGS

class DuckDuckGoSearch:
    def __init__(self):
        self.ddgs = DDGS()

    def search(self, query, region='cn-zh', max_results=10):
        results = list(self.ddgs.text(query, region=region, max_results=max_results))
        return results

    def __call__(self, query, region='cn-zh', max_results=10):
        return self.search(query, region, max_results)

# 创建一个全局实例
search_agent = DuckDuckGoSearch()

if __name__ == "__main__":
    results = search_agent("上海古埃及展览结束没有")
    from pprint import pprint
    pprint(results)