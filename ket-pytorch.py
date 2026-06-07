import torch
import numpy as np
from typing import Callable, Any, Tuple
from ket.qulib import dump_matrix

class KetUnitaryBridge(torch.autograd.Function):
    
    @staticmethod
    def forward(
        ctx: Any, 
        theta_tensor: torch.Tensor, 
        ansatz_function: Callable, 
        num_qubits: int
    ) -> torch.Tensor:
        """Fluxo de IDA (Rápido e Direto)"""
        ctx.save_for_backward(theta_tensor)
        ctx.ansatz_function = ansatz_function
        ctx.num_qubits = num_qubits

        theta: np.ndarray = theta_tensor.detach().numpy()
        matriz_numpy: np.ndarray = np.array(dump_matrix(ansatz_function(theta), num_qubits))

        return torch.tensor(matriz_numpy, dtype=torch.complex128)

    @staticmethod
    def backward(
        ctx: Any, 
        grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None]:
        """Fluxo de VOLTA (Sequencial puro, SEM multiprocessing)"""
        theta_tensor, = ctx.saved_tensors
        ansatz_function: Callable = ctx.ansatz_function
        num_qubits: int = ctx.num_qubits

        theta: np.ndarray = theta_tensor.detach().numpy()
        grad_output_np: np.ndarray = grad_output.detach().numpy()
        grad_theta: torch.Tensor = torch.zeros_like(theta_tensor, dtype=torch.float64)
        
        shift: float = np.pi / 2.0
        
        # Loop sequencial normal: Sem criar processos, sem travar o OS!
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[i] += shift
            theta_minus[i] -= shift
            
            U_plus_np = np.array(dump_matrix(ansatz_function(theta_plus), num_qubits))
            U_minus_np = np.array(dump_matrix(ansatz_function(theta_minus), num_qubits))
            
            dU_theta = (U_plus_np - U_minus_np) / 2.0
            
            # CORREÇÃO AQUI: np.sum em vez de torch.sum, forçado para float
            grad_theta[i] = float(np.sum(grad_output_np.conj() * dU_theta).real)
            
        return grad_theta, None, None