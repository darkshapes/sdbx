import os
import tomllib
import logging
import urllib3

from sdbx.clients.releases import download_asset, get_asset_url, parse_service

class ClientManager:
    def __init__(self, extensions, clients_path):
        self.client_signatures = extensions["clients"]
        self.clients_path = clients_path

        self.http = None
        
        if not self.client_signatures:
            # trigger remote/embedded client or something lol
            return

        self.selected_path = self.validate_clients_installed()
    
    def validate_clients_installed(self):
        first_viable = None
        http = urllib3.PoolManager()

        for client_signature, url in self.client_signatures.items():
            client_path = os.path.join(self.clients_path, os.path.normpath(client_signature))

            if not os.path.exists(client_path):
                logging.info(f"Client {client_signature} not installed, downloading...")
                
                if not self.http:
                    self.http = urllib3.PoolManager()

                namespace, project, service = parse_service(url, client_signature)
                asset_url, _ = get_asset_url(self.http, namespace, project, service=service)
                download_asset(self.http, asset_url, client_path)

            if os.path.exists(os.path.join(client_path, "index.html")) and first_viable is None:
                first_viable = client_path
        
        if first_viable is None:
            raise Exception("No viable clients could be found. Check your installations.")

        return first_viable
    
    def update_clients(self):
        for client_signature, url in self.client_signatures.items():
            client_path = os.path.join(self.clients_path, os.path.normpath(client_signature))

            if not os.path.exists(client_path):
                continue  # skip downloading new clients, that's the job of startup
                
            if not self.http:
                self.http = urllib3.PoolManager()

            namespace, project, service = parse_service(url, client_signature)                
            asset_url, lastmodified = get_asset_url(self.http, namespace, project, service=service)
            if os.path.getmtime(client_path) < lastmodified:
                download_asset(self.http, asset_url, client_path)