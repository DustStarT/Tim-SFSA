import torch
import torch.nn as nn
import numpy as np
from .spline import *


class KANLayer(nn.Module):
    """
    KANLayer class
    

    Attributes:
    -----------
        in_dim: int
            input dimension
        out_dim: int
            output dimension
        size: int
            the number of splines = input dimension * output dimension
        k: int
            the piecewise polynomial order of splines
        grid: 2D torch.float
            grid points
        noises: 2D torch.float
            injected noises to splines at initialization (to break degeneracy)
        coef: 2D torch.tensor
            coefficients of B-spline bases
        scale_base: 1D torch.float
            magnitude of the residual function b(x)
        scale_sp: 1D torch.float
            mangitude of the spline function spline(x)
        base_fun: fun
            residual function b(x)
        mask: 1D torch.float
            mask of spline functions. setting some element of the mask to zero means
            setting the corresponding activation to zero function.
        grid_eps: float in [0,1]
            a hyperparameter used in update_grid_from_samples. When grid_eps = 0, the grid is uniform;
            when grid_eps = 1, the grid is partitioned using percentiles of samples.
            0 < grid_eps < 1 interpolates between the two extremes.
        weight_sharing: 1D tensor int
            allow spline activations to share parameters
        lock_counter: int
            counter how many activation functions are locked (weight sharing)
        lock_id: 1D torch.int
            the id of activation functions that are locked
        device: str
            device
    
    Methods:
    --------
        __init__():
            initialize a KANLayer
        forward():
            forward 
        update_grid_from_samples():
            update grids based on samples' incoming activations
        initialize_grid_from_parent():
            initialize grids from another model
        get_subset():
            get subset of the KANLayer (used for pruning)
        lock():
            lock several activation functions to share parameters
        unlock():
            unlock already locked activation functions
    """

    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.1, scale_base=1.0, scale_sp=1.0,
                 base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=None,
                 sp_trainable=True, sb_trainable=True, device='cpu'):
        """
        initialize a KANLayer

        Args:
        -----
            in_dim : int
                input dimension. Default: 2.
            out_dim : int
                output dimension. Default: 3.
            num : int
                the number of grid intervals = G. Default: 5.
            k : int
                the order of piecewise polynomial. Default: 3.
            noise_scale : float
                the scale of noise injected at initialization. Default: 0.1.
            scale_base : float
                the scale of the residual function b(x). Default: 1.0.
            scale_sp : float
                the scale of the base function spline(x). Default: 1.0.
            base_fun : function
                residual function b(x). Default: torch.nn.SiLU()
            grid_eps : float
                When grid_eps = 0, the grid is uniform; when grid_eps = 1, the grid is partitioned using percentiles
                of samples. 0 < grid_eps < 1 interpolates between the two extremes. Default: 0.02.
            grid_range : list/np.array of shape (2,)
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp is trainable. Default: True.
            sb_trainable : bool
                If true, scale_base is trainable. Default: True.
            device : str
                device

        Returns:
        --------
            self

        Example
        -------
        # >>> model = KANLayer(in_dim=3, out_dim=5)
        # >>> (model.in_dim, model.out_dim)
        (3, 5)
        """
        super(KANLayer, self).__init__()
        # size 
        if grid_range is None:
            grid_range = [-1, 1]
        self.size = size = out_dim * in_dim
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k

        # shape: (size, num)
        grid = torch.einsum('i,j->ij', torch.ones(size, device=device),
                                 torch.linspace(grid_range[0], grid_range[1], steps=num + 1, device=device))
        self.register_buffer("grid", grid)
        noises = (torch.rand(size, self.grid.shape[1]) - 1 / 2) * noise_scale / num
        noises = noises.to(device)
        # shape: (size, coef)
        self.coef = torch.nn.Parameter(curve2coef(self.grid, noises, self.grid, k, device))
        if isinstance(scale_base, float):
            self.scale_base = torch.nn.Parameter(
                torch.ones(size, device=device) * scale_base).requires_grad_(sb_trainable)  # make scale trainable
        else:
            self.scale_base = torch.nn.Parameter(scale_base).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(
            torch.ones(size, device=device) * scale_sp).requires_grad_(sp_trainable)  # make scale trainable
            
        # 确保base_fun是一个nn.Module实例
        if isinstance(base_fun, torch.nn.Module):
            self.base_fun = base_fun
        else:
            # 如果不是nn.Module但是一个可调用对象，尝试使用
            try:
                test_input = torch.zeros(1, 1, 1, device=device)
                base_fun(test_input)  # 测试是否可调用
                self.base_fun = base_fun
            except:
                # 如果无法使用，回退到默认激活函数
                print("Warning: base_fun不是可调用的激活函数，使用默认的SiLU")
                self.base_fun = torch.nn.SiLU()

        self.mask = torch.nn.Parameter(torch.ones(size, device=device)).requires_grad_(False)
        self.grid_eps = grid_eps
        self.weight_sharing = torch.arange(size, device=device)
        self.lock_counter = 0
        self.lock_id = torch.zeros(size, device=device)
        self.device = device

    def forward(self, x):
        """
        KANLayer forward given input x

        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            y : 2D torch.float
                outputs, shape (number of samples, output dimension)
            preacts : 3D torch.float
                fan out x into activations, shape (number of sampels, output dimension, input dimension)
            postacts : 3D torch.float
                the outputs of activation functions with preacts as inputs
            postspline : 3D torch.float
                the outputs of spline functions with preacts as inputs

        Example
        -------
        # >>> model = KANLayer(in_dim=3, out_dim=5)
        # >>> x = torch.normal(0,1,size=(100,3))
        # >>> y, preacts, postacts, postspline = model(x)
        # >>> (y.shape, preacts.shape, postacts.shape, postspline.shape)
        (torch.Size([100, 5]), torch.Size([100, 5, 3]), torch.Size([100, 5, 3]), torch.Size([100, 5, 3]))
        """
        B = x.shape[0]
        x_fan_out = x[:, None, :].repeat(1, self.out_dim, 1)  # shape [B, out, in]

        # solve x = (preact)
        preacts = x_fan_out  # shape [B, out, in]

        # compute spline(x)
        grid_all = self.grid[self.weight_sharing]
        coef_all = self.coef[self.weight_sharing]
        postspline = coef2curve(x_eval=preacts.reshape(B, self.out_dim * self.in_dim).transpose(1, 0),
                                grid=grid_all, coef=coef_all, k=self.k, device=self.device)
        postspline = postspline.transpose(1, 0).reshape(B, self.out_dim, self.in_dim)

        # 计算基础激活函数
        base = self.base_fun(preacts)
        
        # 合并结果
        scale_base_reshape = self.scale_base[self.weight_sharing].reshape(1, self.out_dim, self.in_dim)
        scale_sp_reshape = self.scale_sp[self.weight_sharing].reshape(1, self.out_dim, self.in_dim)
        mask_reshape = self.mask[self.weight_sharing].reshape(1, self.out_dim, self.in_dim)

        # combine base and spline
        postacts = scale_sp_reshape * postspline + scale_base_reshape * base
        postacts = mask_reshape * postacts

        # sum over in_dim
        y = torch.sum(postacts, dim=2)

        return y, preacts, postacts, postspline

    def update_grid_from_samples(self, x):
        """
        update grid from samples

        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            None

        Example
        -------
        # >>> model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        # >>> print(model.grid.data)
        # >>> x = torch.linspace(-3,3,steps=100)[:,None]
        # >>> model.update_grid_from_samples(x)
        # >>> print(model.grid.data)
        tensor([[-1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000]])
        tensor([[-3.0002, -1.7882, -0.5763,  0.6357,  1.8476,  3.0002]])
        """
        batch = x.shape[0]
        x = torch.einsum(
            'ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(1, 0)
        x_pos = torch.sort(x, dim=1)[0]
        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k, device=self.device)
        num_interval = self.grid.shape[1] - 1
        ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
        grid_adaptive = x_pos[:, ids]
        margin = 0.01
        grid_uniform = torch.cat([
            grid_adaptive[:, [0]] - margin +
            (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]] + 2 * margin) * a for a in np.linspace(
                0, 1, num=self.grid.shape[1])], dim=1)
        self.grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k, device=self.device)

    def initialize_grid_from_parent(self, parent, x):
        """
        update grid from a parent KANLayer & samples

        Args:
        -----
            parent : KANLayer
                a parent KANLayer (whose grid is usually coarser than the current model)
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            None

        Example
        -------
        # >>> batch = 100
        # >>> parent_model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        # >>> print(parent_model.grid.data)
        # >>> model = KANLayer(in_dim=1, out_dim=1, num=10, k=3)
        # >>> x = torch.normal(0,1,size=(batch, 1))
        # >>> model.initialize_grid_from_parent(parent_model, x)
        # >>> print(model.grid.data)
        tensor([[-1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000]])
        tensor([[-1.0000, -0.8000, -0.6000, -0.4000, -0.2000,  0.0000,  0.2000,  0.4000,
          0.6000,  0.8000,  1.0000]])
        """
        batch = x.shape[0]
        # preacts: shape (batch, in_dim) => shape (size, batch) (size = out_dim * in_dim)
        x_eval = torch.einsum('ij,k->ikj', x, torch.ones(
            self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(1, 0)
        x_pos = parent.grid
        sp2 = KANLayer(in_dim=1, out_dim=self.size, k=1, num=x_pos.shape[1] - 1, scale_base=0., device=self.device)
        sp2.coef.data = curve2coef(sp2.grid, x_pos, sp2.grid, k=1, device=self.device)
        y_eval = coef2curve(x_eval, parent.grid, parent.coef, parent.k, device=self.device)
        percentile = torch.linspace(-1, 1, self.num + 1).to(self.device)
        self.grid = sp2(percentile.unsqueeze(dim=1))[0].permute(1, 0)
        self.coef.data = curve2coef(x_eval, y_eval, self.grid, self.k, self.device)

    def get_subset(self, in_id, out_id):
        """
        get subset of the KANLayer (used for pruning)
        
        Args:
        -----
            in_id : list/array
                indices of input neurons
            out_id : list/array
                indices of output neurons
                
        Returns:
        --------
            spb : KANLayer
                subset of the KANLayer

        Example
        -------
        # >>> model = KANLayer(in_dim=3, out_dim=5)
        # >>> sub = model.get_subset([0,1], [1,3])
        # >>> (sub.in_dim, sub.out_dim)
        (2, 2)
        # >>> sub.mask
        tensor([1., 1., 1., 1.])
        """
        # 直接使用当前的base_fun创建新的KANLayer
        spb = KANLayer(len(in_id), len(out_id), self.num, self.k, 
                      base_fun=self.base_fun, device=self.device)
        
        # 复制其他参数
        spb.grid = self.grid.reshape(
            self.out_dim, self.in_dim, spb.num + 1)[out_id][:, in_id].reshape(-1, spb.num + 1)

        out_by_in = self.out_dim * self.in_dim
        coef_k = self.coef.shape[1]
        mask_sub = torch.zeros(len(out_id), len(in_id), device=self.device)
        scale_sp_sub = torch.zeros(len(out_id), len(in_id), device=self.device)
        scale_base_sub = torch.zeros(len(out_id), len(in_id), device=self.device)
        coef_sub = torch.zeros(len(out_id), len(in_id), coef_k, device=self.device)
        weight_sharing_sub = torch.zeros(len(out_id), len(in_id), device=self.device).long()

        for i, ii in enumerate(in_id):
            for j, jj in enumerate(out_id):
                mask_sub[j, i] = self.mask[jj * self.in_dim + ii]
                scale_sp_sub[j, i] = self.scale_sp[jj * self.in_dim + ii]
                scale_base_sub[j, i] = self.scale_base[jj * self.in_dim + ii]
                coef_sub[j, i, :] = self.coef[jj * self.in_dim + ii]
                weight_sharing_sub[j, i] = i + j * len(in_id)

        spb.mask = nn.Parameter(mask_sub.reshape(-1).detach()).requires_grad_(False)
        spb.scale_sp = nn.Parameter(scale_sp_sub.reshape(-1).detach())
        spb.scale_base = nn.Parameter(scale_base_sub.reshape(-1).detach())
        spb.coef = nn.Parameter(coef_sub.reshape(-1, coef_k).detach())
        spb.weight_sharing = weight_sharing_sub.reshape(-1)

        return spb

    def lock(self, ids):
        """
        lock activation functions to share parameters based on ids

        Args:
        -----
            ids : list
                list of ids of activation functions

        Returns:
        --------
            None

        Example
        -------
        # >>> model = KANLayer(in_dim=3, out_dim=3, num=5, k=3)
        # >>> print(model.weight_sharing.reshape(3,3))
        # >>> model.lock([[0,0],[1,2],[2,1]]) # set (0,0),(1,2),(2,1) functions to be the same
        # >>> print(model.weight_sharing.reshape(3,3))
        tensor([[0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]])
        tensor([[0, 1, 2],
                [3, 4, 0],
                [6, 0, 8]])
        """
        self.lock_counter += 1
        # ids: [[i1,j1],[i2,j2],[i3,j3],...]
        for i in range(len(ids)):
            if i != 0:
                self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] = ids[0][1] * self.in_dim + ids[0][0]
            self.lock_id[ids[i][1] * self.in_dim + ids[i][0]] = self.lock_counter

    def unlock(self, ids):
        """
        unlock activation functions

        Args:
        -----
            ids : list
                list of ids of activation functions

        Returns:
        --------
            None

        Example
        -------
        # >>> model = KANLayer(in_dim=3, out_dim=3, num=5, k=3)
        # >>> model.lock([[0,0],[1,2],[2,1]]) # set (0,0),(1,2),(2,1) functions to be the same
        # >>> print(model.weight_sharing.reshape(3,3))
        # >>> model.unlock([[0,0],[1,2],[2,1]]) # unlock the locked functions
        # >>> print(model.weight_sharing.reshape(3,3))
        tensor([[0, 1, 2],
                [3, 4, 0],
                [6, 0, 8]])
        tensor([[0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]])
        """
        # check ids are locked
        num = len(ids)
        locked = True
        for i in range(num):
            locked *= (self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] ==
                       self.weight_sharing[ids[0][1] * self.in_dim + ids[0][0]])
        if not locked:
            print("they are not locked. unlock failed.")
            return 0
        for i in range(len(ids)):
            self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] = ids[i][1] * self.in_dim + ids[i][0]
            self.lock_id[ids[i][1] * self.in_dim + ids[i][0]] = 0
        self.lock_counter -= 1
