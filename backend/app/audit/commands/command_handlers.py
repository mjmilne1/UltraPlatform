"""
CQRS Command Handlers
Commands that modify state (writes)
"""

from typing import Dict
from datetime import datetime
import uuid

class CreateClientCommand:
    """Command to create new client"""
    def __init__(self, client_data: Dict, user_id: str):
        self.command_id = str(uuid.uuid4())
        self.client_data = client_data
        self.user_id = user_id
        self.timestamp = datetime.now()

class UpdateAccountCommand:
    """Command to update account"""
    def __init__(self, account_id: str, updates: Dict, user_id: str):
        self.command_id = str(uuid.uuid4())
        self.account_id = account_id
        self.updates = updates
        self.user_id = user_id
        self.timestamp = datetime.now()

class CommandHandler:
    """
    CQRS Command Handler
    
    All state-changing operations go through commands
    Commands are:
    - Validated
    - Logged in audit trail
    - May require approval (maker-checker)
    - Event sourced
    """
    
    def __init__(self, audit_service, event_store):
        self.audit = audit_service
        self.event_store = event_store
    
    async def handle_create_client(
        self,
        command: CreateClientCommand,
        requires_approval: bool = True
    ) -> Dict:
        """Handle client creation command"""
        
        if requires_approval:
            # Create approval request
            request = await self.audit.create_approval_request(
                maker_id=command.user_id,
                maker_email="maker@example.com",  # Would lookup
                entity_type="CLIENT",
                entity_id=command.command_id,
                change_type="CREATE",
                proposed_changes=command.client_data,
                reason="New client onboarding"
            )
            
            return {
                "status": "pending_approval",
                "request_id": request.request_id,
                "command_id": command.command_id
            }
        else:
            # Execute directly
            client_id = await self._execute_create_client(command)
            
            # Log in audit trail
            await self.audit.log_change(
                user_id=command.user_id,
                user_email="user@example.com",
                user_role="admin",
                entity_type="CLIENT",
                entity_id=client_id,
                change_type="CREATE",
                new_value=command.client_data,
                reason="Direct creation"
            )
            
            return {
                "status": "created",
                "client_id": client_id
            }
    
    async def handle_update_account(
        self,
        command: UpdateAccountCommand,
        requires_approval: bool = True
    ) -> Dict:
        """Handle account update command"""
        
        if requires_approval:
            request = await self.audit.create_approval_request(
                maker_id=command.user_id,
                maker_email="maker@example.com",
                entity_type="ACCOUNT",
                entity_id=command.account_id,
                change_type="UPDATE",
                proposed_changes=command.updates,
                reason="Account update"
            )
            
            return {
                "status": "pending_approval",
                "request_id": request.request_id
            }
        else:
            await self._execute_update_account(command)
            
            await self.audit.log_change(
                user_id=command.user_id,
                user_email="user@example.com",
                user_role="admin",
                entity_type="ACCOUNT",
                entity_id=command.account_id,
                change_type="UPDATE",
                new_value=command.updates,
                reason="Direct update"
            )
            
            return {
                "status": "updated",
                "account_id": command.account_id
            }
    
    async def _execute_create_client(self, command: CreateClientCommand) -> str:
        """Execute client creation"""
        client_id = str(uuid.uuid4())
        # Would create in database
        return client_id
    
    async def _execute_update_account(self, command: UpdateAccountCommand):
        """Execute account update"""
        # Would update in database
        pass
