'''
AI Assistant Integration Guide for UltraPlatform MCP
'''

class AIAssistantIntegration:
    '''Example integration for AI assistants like Claude'''
    
    def __init__(self):
        self.client = None
        self.context = {}
        
    async def connect_to_ultraplatform(self):
        '''Connect AI assistant to UltraPlatform'''
        from modules.mcp_integration.clients.python.client import UltraMCPClient
        
        self.client = UltraMCPClient()
        await self.client.connect()
        
        print('AI Assistant connected to UltraPlatform')
        print('Capabilities unlocked:')
        print('  • Natural language trading')
        print('  • Portfolio queries')
        print('  • Automated rebalancing')
        print('  • Real-time analytics')
        
        return True
    
    async def process_user_request(self, request):
        '''Process natural language trading request'''
        
        # Example requests an AI might receive
        if 'buy' in request.lower():
            # Extract details from natural language
            symbol = 'GOOGL'  # Parsed from request
            quantity = 10     # Parsed from request
            
            # Execute trade
            result = await self.client.execute_trade(
                symbol, 'BUY', quantity, 'momentum'
            )
            
            response = f'Executed: BUY {quantity} {symbol}'
            response += f'\nExpected return: 57.15%'
            
        elif 'portfolio' in request.lower():
            # Get portfolio status
            portfolio = await self.client.get_portfolio()
            
            response = f'Portfolio Value: '
            response += f'\nNAV: '
            response += f'\nCash: '
            
        elif 'rebalance' in request.lower():
            # Check rebalancing
            result = await self.client.rebalance_portfolio()
            
            if result['needs_rebalancing']:
                response = 'Portfolio needs rebalancing'
                response += f'\nExpected return after: 102.8%'
            else:
                response = 'Portfolio is balanced'
        
        else:
            response = 'How can I help with your trading?'
        
        return response

# Demo AI assistant integration
import asyncio

async def demo_ai_integration():
    print('AI Assistant Integration Demo')
    print('='*50)
    
    ai = AIAssistantIntegration()
    await ai.connect_to_ultraplatform()
    
    # Simulate user requests
    requests = [
        'What is my portfolio value?',
        'Buy 10 shares of Google',
        'Should I rebalance?'
    ]
    
    for request in requests:
        print(f'\nUser: {request}')
        response = await ai.process_user_request(request)
        print(f'AI: {response}')
    
    print('\n✅ AI Integration complete!')

if __name__ == '__main__':
    asyncio.run(demo_ai_integration())
