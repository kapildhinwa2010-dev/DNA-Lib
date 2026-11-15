import pytest
import torch
from dna_lib import DNALib, create_dna_lib

class TestDNALib:
    def test_init(self):
        m = DNALib(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = DNALib(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_dna_lib()
        assert isinstance(m, DNALib)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
