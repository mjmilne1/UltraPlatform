from datetime import datetime

class DataCatalog:
    def __init__(self):
        self.catalog_entries = {}
        self.discovery_engine = DiscoveryEngine()
        self.search_engine = SearchEngine()
        self._initialize_catalog()
        
    def _initialize_catalog(self):
        self.register_product({'id': 'dp-trading-signals-001', 'name': 'AI Trading Signals', 'domain': 'trading', 'type': 'stream', 'description': 'Real-time AI signals with 57% returns', 'owner': 'trading-team', 'tags': ['ai', 'ml', 'signals', 'trading'], 'quality': 'platinum', 'access': 'api://trading/signals'})
        self.register_product({'id': 'dp-portfolio-nav-001', 'name': 'Portfolio NAV', 'domain': 'portfolio', 'type': 'snapshot', 'description': 'Real-time NAV calculation', 'owner': 'portfolio-team', 'tags': ['nav', 'portfolio', 'valuation'], 'quality': 'gold', 'access': 'api://portfolio/nav'})
        self.register_product({'id': 'dp-risk-metrics-001', 'name': 'Risk Metrics', 'domain': 'risk', 'type': 'metrics', 'description': 'VaR and Sharpe ratio', 'owner': 'risk-team', 'tags': ['risk', 'var', 'sharpe'], 'quality': 'gold', 'access': 'api://risk/metrics'})
        self.register_product({'id': 'dp-market-prices-001', 'name': 'Market Prices', 'domain': 'market', 'type': 'stream', 'description': 'Sub-second price updates', 'owner': 'market-team', 'tags': ['prices', 'market', 'feed'], 'quality': 'silver', 'access': 'ws://market/prices'})
        self.register_product({'id': 'dp-analytics-001', 'name': 'Performance Analytics', 'domain': 'analytics', 'type': 'report', 'description': 'Daily P&L and performance', 'owner': 'analytics-team', 'tags': ['analytics', 'pnl', 'reporting'], 'quality': 'gold', 'access': 'api://analytics/performance'})
    
    def register_product(self, metadata):
        product_id = metadata['id']
        self.catalog_entries[product_id] = metadata
        self.search_engine.index_product(metadata)
        
    def discover_products(self, filters=None):
        if not filters:
            return list(self.catalog_entries.values())
        results = []
        for pid, product in self.catalog_entries.items():
            match = True
            if 'domain' in filters and product['domain'] != filters['domain']:
                match = False
            if 'quality' in filters and product['quality'] != filters['quality']:
                match = False
            if match:
                results.append(product)
        return results
    
    def search_catalog(self, query):
        return self.search_engine.search(query, self.catalog_entries)

class DiscoveryEngine:
    def discover(self, catalog, filters=None):
        if not filters:
            return list(catalog.values())
        return []

class SearchEngine:
    def __init__(self):
        self.index = {}
        
    def index_product(self, product):
        product_id = product['id']
        searchable = ' '.join([product['name'].lower(), product['description'].lower(), ' '.join(product['tags'])])
        for word in searchable.split():
            if word not in self.index:
                self.index[word] = []
            if product_id not in self.index[word]:
                self.index[word].append(product_id)
    
    def search(self, query, catalog):
        query_words = query.lower().split()
        matching_ids = set()
        for word in query_words:
            if word in self.index:
                matching_ids.update(self.index[word])
        results = []
        for pid in matching_ids:
            if pid in catalog:
                results.append(catalog[pid])
        return results

class CatalogInterface:
    def __init__(self, catalog):
        self.catalog = catalog
        
    def display_catalog(self):
        print('DATA PRODUCT CATALOG')
        print('='*70)
        print('Total Products: ' + str(len(self.catalog.catalog_entries)))
        for domain in ['trading', 'portfolio', 'risk', 'market', 'analytics']:
            products = self.catalog.discover_products({'domain': domain})
            if products:
                print('\n' + domain.upper() + ' DOMAIN:')
                print('-'*40)
                for product in products:
                    print('  Name: ' + product['name'])
                    print('  ID: ' + product['id'])
                    print('  Quality: ' + product['quality'])
                    print('  Access: ' + product['access'])
    
    def search_interface(self, query):
        print('\nSEARCH RESULTS for "' + query + '":')
        print('-'*40)
        results = self.catalog.search_catalog(query)
        if results:
            for product in results:
                print('  • ' + product['name'] + ' (' + product['domain'] + ')')
        else:
            print('  No products found')
        return results

if __name__ == '__main__':
    print('DATA DISCOVERY AND CATALOG')
    print('='*70)
    catalog = DataCatalog()
    interface = CatalogInterface(catalog)
    interface.display_catalog()
    print('\nSEARCH TESTS:')
    interface.search_interface('trading signals')
    interface.search_interface('nav')
    print('\nGOLD QUALITY PRODUCTS:')
    print('-'*40)
    gold = catalog.discover_products({'quality': 'gold'})
    for product in gold:
        print('  • ' + product['name'])
    print('\nData Catalog operational!')
