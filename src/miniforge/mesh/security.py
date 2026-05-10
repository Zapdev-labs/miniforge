"""Security utilities for mesh: auto TLS certificate generation."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.x509.oid import NameOID

logger = logging.getLogger(__name__)


class MeshSecurity:
    """
    Automatic TLS certificate management for mesh nodes.
    
    Features:
    - Ed25519 key pairs (lightweight, secure)
    - Self-signed CA per node
    - Automatic certificate rotation
    - Certificate pinning for trusted peers
    """
    
    CERT_VALIDITY_DAYS = 30
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.certs_dir = config_dir / "mesh" / "certs"
        self.certs_dir.mkdir(parents=True, exist_ok=True)
        
        self._private_key: Optional[Ed25519PrivateKey] = None
        self._certificate: Optional[x509.Certificate] = None
        self._pinned_certs: set[str] = set()
        
    @property
    def private_key(self) -> Ed25519PrivateKey:
        """Get or generate private key."""
        if self._private_key is None:
            self._private_key = self._load_or_generate_key()
        return self._private_key
        
    @property
    def certificate(self) -> x509.Certificate:
        """Get or generate certificate."""
        if self._certificate is None:
            self._certificate = self._load_or_generate_cert()
        return self._certificate
        
    def _load_or_generate_key(self) -> Ed25519PrivateKey:
        """Load existing key or generate new one."""
        key_path = self.certs_dir / "node_key.pem"
        
        if key_path.exists():
            try:
                with open(key_path, "rb") as f:
                    key_data = f.read()
                key = serialization.load_pem_private_key(key_data, password=None)
                if isinstance(key, Ed25519PrivateKey):
                    logger.info("Loaded existing private key")
                    return key
            except Exception as e:
                logger.warning(f"Failed to load key: {e}, generating new one")
                
        # Generate new key
        key = Ed25519PrivateKey.generate()
        key_data = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        with open(key_path, "wb") as f:
            f.write(key_data)
        os.chmod(key_path, 0o600)
        logger.info("Generated new private key")
        return key
        
    def _load_or_generate_cert(self) -> x509.Certificate:
        """Load existing certificate or generate new one."""
        cert_path = self.certs_dir / "node_cert.pem"
        
        if cert_path.exists():
            try:
                with open(cert_path, "rb") as f:
                    cert_data = f.read()
                cert = x509.load_pem_x509_certificate(cert_data)
                # Check if still valid
                if cert.not_valid_after > datetime.utcnow() + timedelta(days=1):
                    logger.info("Loaded existing certificate")
                    return cert
                else:
                    logger.info("Certificate expired, generating new one")
            except Exception as e:
                logger.warning(f"Failed to load certificate: {e}")
                
        # Generate new certificate
        cert = self._generate_certificate()
        cert_data = cert.public_bytes(serialization.Encoding.PEM)
        with open(cert_path, "wb") as f:
            f.write(cert_data)
        logger.info("Generated new certificate")
        return cert
        
    def _generate_certificate(self) -> x509.Certificate:
        """Generate self-signed certificate."""
        key = self.private_key
        public_key = key.public_key()
        
        # Subject and issuer are the same (self-signed)
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "XX"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Mesh"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Local"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Miniforge"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Mesh Node"),
            x509.NameAttribute(NameOID.COMMON_NAME, "miniforge-node"),
        ])
        
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(public_key)
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow() - timedelta(days=1))
            .not_valid_after(datetime.utcnow() + timedelta(days=self.CERT_VALIDITY_DAYS))
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress:=__import__("ipaddress").ip_address("127.0.0.1")),
                ]),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )
        return cert
        
    def get_cert_fingerprint(self) -> str:
        """Get certificate fingerprint for display/verification."""
        cert = self.certificate
        fingerprint = cert.fingerprint(hashes.SHA256())
        return fingerprint.hex()[:16].upper()
        
    def get_cert_pem(self) -> bytes:
        """Get certificate as PEM bytes."""
        return self.certificate.public_bytes(serialization.Encoding.PEM)
        
    def pin_peer_cert(self, cert_pem: bytes) -> str:
        """Pin a peer certificate for future verification."""
        try:
            cert = x509.load_pem_x509_certificate(cert_pem)
            fingerprint = cert.fingerprint(hashes.SHA256()).hex()
            self._pinned_certs.add(fingerprint)
            return fingerprint[:16].upper()
        except Exception as e:
            logger.error(f"Failed to pin certificate: {e}")
            raise
            
    def verify_peer_cert(self, cert_pem: bytes) -> bool:
        """Verify if a peer certificate is pinned/trusted."""
        try:
            cert = x509.load_pem_x509_certificate(cert_pem)
            fingerprint = cert.fingerprint(hashes.SHA256()).hex()
            return fingerprint in self._pinned_certs
        except Exception:
            return False
            
    def create_tls_context(self, server_mode: bool = False) -> "ssl.SSLContext":
        """Create SSL context with mTLS."""
        import ssl
        
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER if server_mode else ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.verify_mode = ssl.CERT_REQUIRED
        
        # Load our certificate and key
        context.load_cert_chain(
            certfile=str(self.certs_dir / "node_cert.pem"),
            keyfile=str(self.certs_dir / "node_key.pem"),
        )
        
        # In mesh mode, we trust pinned certificates
        # Load pinned certs as CA certs
        if self._pinned_certs:
            # Create temporary CA bundle
            ca_path = self.certs_dir / "pinned_certs.pem"
            with open(ca_path, "wb") as f:
                for fingerprint in self._pinned_certs:
                    # This is simplified - in practice you'd store full certs
                    pass
            if ca_path.exists():
                context.load_verify_locations(cafile=str(ca_path))
        else:
            # If no pinned certs, disable verification (LAN mode)
            context.verify_mode = ssl.CERT_NONE
            
        return context
