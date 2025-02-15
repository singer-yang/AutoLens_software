""" Complex wave class. We have to use float64 precision.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.fft import *
import torch.nn.functional as nnf
import torchvision.transforms.functional as F
import pickle
from torchvision.utils import save_image

from .basics import *

# ===================================
# Complex wave field
# ===================================
class ComplexWave(DeepObj):
    def __init__(self, u=None, wvln=0.550, z=0., phy_size=[4., 4.], valid_phy_size=None, res=[1024,1024], device=DEVICE):
        """ Complex wave field class.

        Args:
            amp (_type_, optional): _description_. Defaults to None.
            phi (_type_, optional): _description_. Defaults to None.
            wvln (int, optional): wvln. Defaults to 550.
            size (list, optional): physical size of a field, in [mm].
            res (list, optional): discrete resolution of a field. Defaults to [128,128].

            res
            phy_size
            wvln: [um]
            k
            
            u: can be either [B, 1, H, W] or [H, W]
            x: [H, W]
            y: [H, W]
            z: [H, W]
        """
        super(ComplexWave, self).__init__()

        # Wave field has shape of [N, 1, H, W] for batch processing
        if u is not None:
            self.u = u if torch.is_tensor(u) else torch.from_numpy(u)
            if not self.u.is_complex():
                self.u = self.u.to(torch.complex64)
            
            if len(u.shape) == 2:   # [H, W]
                self.u = u.unsqueeze(0).unsqueeze(0)
            elif len(self.u.shape) == 3:    # [1, H, W]
                self.u = self.u.unsqueeze(0)
            
            self.res = self.u.shape[-2:]

        else:
            amp = torch.zeros(res).unsqueeze(0).unsqueeze(0)
            phi = torch.zeros(res).unsqueeze(0).unsqueeze(0)
            self.u = amp + 1j * phi
            self.res = res        

        # Other paramters
        assert wvln > 0.1 and wvln < 1, 'wvln unit should be [um].'
        self.wvln = wvln # wvln, store in [um]
        self.k = 2 * np.pi / (self.wvln * 1e-3) # distance unit [mm]
        self.phy_size = np.array(phy_size)  # physical size with padding, in [mm]
        self.valid_phy_size = self.phy_size if valid_phy_size is None else np.array(valid_phy_size) # physical size without padding, in [mm]
        
        assert phy_size[0] / self.res[0] == phy_size[1] / self.res[1], 'Wrong pixel size.'
        self.ps = phy_size[0] / self.res[0] # pixel size, float value

        self.x, self.y = self.gen_xy_grid()
        self.z = torch.full_like(self.x, z)

        # self.float16()
        self.to(device)
        

    def load_img(self, img):
        """ Use the pixel value of an image/batch as the amplitute.

            TODO: test this function.

        Args:
            img (ndarray or tensor): shape [H, W] or [B, C, H, W].
        """
        if img.dtype == 'uint8':
            img = img/255.

        if torch.is_tensor(img):
            amp = torch.sqrt(img)
        else:
            amp = torch.sqrt(torch.from_numpy(img/255.))
        
        phi = torch.zeros_like(amp)
        u = amp + 1j * phi
        self.u = u.to(self.device)
        self.res = self.u.shape


    def load_pkl(self, data_path):
        with open(data_path, "rb") as tf:
            wave_data = pickle.load(tf)
            tf.close()

        amp = wave_data['amp']
        phi = wave_data['phi']
        self.u = amp * torch.exp(1j * phi)
        self.x = wave_data['x']
        self.y = wave_data['y']
        self.wvln = wave_data['wvln']
        self.phy_size = wave_data['phy_size']
        self.valid_phy_size = wave_data['valid_phy_size']
        self.res = self.x.shape

        self.to(self.device)


    def save_data(self, save_path='./test.pkl'):
        data = {
            'amp': self.u.cpu().abs(),
            'phi': torch.angle(self.u.cpu()),
            'x': self.x.cpu(),
            'y': self.y.cpu(),
            'wvln': self.wvln,
            'phy_size': self.phy_size,
            'valid_phy_size': self.valid_phy_size
            }

        with open(save_path, 'wb') as tf:
            pickle.dump(data, tf)
            tf.close()

        # cv.imwrite(f'{save_path[:-4]}.png', self.u.cpu().abs()**2)
        intensity = self.u.cpu().abs()**2
        save_image(intensity, f'{save_path[:-4]}.png', normalize=True)
        save_image(torch.abs(self.u.cpu()), f'{save_path[:-4]}_amp.jpg', normalize=True)
        save_image(torch.angle(self.u.cpu()), f'{save_path[:-4]}_phase.jpg', normalize=True)


    # =============================================
    # Operation
    # =============================================

    def float16(self):
        raise Exception('Useless.')
        self.u = self.u.to(torch.complex32)
        self.x = self.x.to(torch.float16)
        self.y = self.y.to(torch.float16)
        self.z = self.z.to(torch.float16)
        return self
    
    def flip(self):
        self.u = torch.flip(self.u, [-1,-2])
        self.x = torch.flip(self.x, [-1,-2])
        self.y = torch.flip(self.y, [-1,-2])
        self.z = torch.flip(self.z, [-1,-2])

        return self

    def prop(self, prop_dist, n = 1.):
        """ Propagate the field by distance z. Can only propagate planar wave. 

            The definition of near-field and far-field depends on the specific problem we want to solve. For diffraction simulation, typically we use Fresnel number to determine the propagation method. In Electro-magnetic applications and fiber optics, the definition is different.
        
            This function now supports batch operation, but only for mono-channel field. Shape of [B, 1, H, W].

            Reference: 
                1, https://spie.org/samples/PM103.pdf
                2, "Non-approximated Rayleigh Sommerfeld diffraction integral: advantages and disadvantages in the propagation of complex wave fields"

            Different methods:
                1, Rayleigh-Sommerfeld Diffraction Formula
                    pros: (a) intermediate and short distance, (b) non-paraxial, (c) ...
                    cons: (a) complexity, (b) scalar wave only, (c) not suitable for long distance, (d) ...
                2, Fresnel diffraction
                3, Fraunhofer diffraction
                4, Finite Difference Time Domain (FDTD)
                5, Beam Propagation Method (BPM)
                6, Angular Spectrum Method (ASM)
                7, Green's function method
                8, Split-step Fourier method

        Args:
            z (float): propagation distance, unit [mm].
        """
        wvln = self.wvln * 1e-3 # [um] -> [mm]
        valid_phy_size = self.valid_phy_size
        if torch.is_tensor(prop_dist):
            prop_dist = prop_dist.item()

        # Determine which propagation method to use by Fresnel number
        num_fresnel = valid_phy_size[0] * valid_phy_size[1] / (wvln * np.abs(prop_dist)) if prop_dist != 0 else 0
        if prop_dist < DELTA:
            # Zero distance, do nothing
            pass

        elif prop_dist < wvln / 2:
            # Sub-wavelength distance: EM method
            raise Exception('EM method is not implemented.')

        # elif num_fresnel < 0.01:
        #     # Far-field diffraction: Fraunhofer approximation 
        #     # Another criterion: z >> k * r2^2 / 2 
        #     self.u = FraunhoferDiffraction(self.u, z=prop_dist, wvln=self.wvln, ps=self.ps, n=n)

        # elif num_fresnel < 10:
        #     # Near-field diffraction: Fresnel approximation
        #     # Require: (1) distance is larger than wvln, (2) paraxial condition
        #     # Another criterion: z >> (k * |r1 + r2|^4 / 8) ^ (1/3)
        #     self.u = FresnelDiffraction(self.u, z=prop_dist, wvln=self.wvln, ps=self.ps, n=n)

        else:
            # Super short distance: Rayleigh-Sommerfeld
            prop_dist_min = self.Nyquist_zmin()
            # assert prop_dist_min < np.abs(prop_dist), 'Propagation distance is too short.'
            if np.abs(prop_dist) < prop_dist_min:
                print("Propagation distance is too short.")
            # self.u = RayleighSommerfeldIntegral(self.u, self.x, self.y, z=prop_dist, wvln=self.wvln, n=n)
            self.u = AngularSpectrumMethod(self.u, z=prop_dist, wvln=self.wvln, ps=self.ps, n=n)
        
        self.z += prop_dist
        return self


    def prop_to(self, z, n=1):
        """ Propagate the field to plane z.

        Args:
            z (float): destination plane z coordinate.
        """
        if torch.is_tensor(z):
            z = z.item()
        prop_dist = z - self.z[0, 0].item()

        self.prop(prop_dist, n=n)
        return self


    def resize(self, size, mode=1):
        self.res = size
        self.ps = self.phy_size[0] / size[0]
        # self.res = [scale * res for res in self.res]
        # self.ps /= scale

        if mode == 1:
            amp_inv = 1 / (self.u.abs() + DELTA)
            amp_inv = nnf.interpolate(amp_inv.unsqueeze(0).unsqueeze(0), size=self.res, mode='bilinear').squeeze(0).squeeze(0)
            amp = 1 / amp_inv
            phi = torch.angle(self.u)
            phi = nnf.interpolate(phi.unsqueeze(0).unsqueeze(0), size=self.res, mode='bilinear').squeeze(0).squeeze(0)
            u = amp * torch.cos(phi) + 1j * amp * torch.sin(phi)
            self.u = u
        elif mode == 2: # Interpolate by amp and phi
            amp = self.u.abs()
            phi = torch.angle(self.u)
            if len(amp.shape) == 2:
                amp = nnf.interpolate(amp.unsqueeze(0).unsqueeze(0), size=self.res, mode='bilinear').squeeze(0).squeeze(0)
                phi = nnf.interpolate(phi.unsqueeze(0).unsqueeze(0), size=self.res, mode='bilinear').squeeze(0).squeeze(0)
            elif len(amp.shape) == 4:
                amp = nnf.interpolate(amp, size=self.res, mode='bilinear')
                phi = nnf.interpolate(phi, size=self.res, mode='bilinear')
            
            u = amp * torch.cos(phi) + 1j * amp * torch.sin(phi)
            self.u = u
        else:   # Interpolate by real and imag
            real = self.u.real
            real = nnf.interpolate(real.unsqueeze(0).unsqueeze(0), size=self.res, mode='bilinear').squeeze(0).squeeze(0)
            imag = self.u.imag
            imag = nnf.interpolate(imag.unsqueeze(0).unsqueeze(0), size=self.res, mode='bilinear').squeeze(0).squeeze(0)
            u = real + 1j * imag
            self.u = u


        self.x = nnf.interpolate(self.x.unsqueeze(0).unsqueeze(0), size=self.res, mode='bilinear').squeeze(0).squeeze(0)
        self.y = nnf.interpolate(self.y.unsqueeze(0).unsqueeze(0), size=self.res, mode='bilinear').squeeze(0).squeeze(0)
        self.z = nnf.interpolate(self.z.unsqueeze(0).unsqueeze(0), size=self.res, mode='bilinear').squeeze(0).squeeze(0)
        

    # =============================================
    # 
    # =============================================
    def gen_xy_grid(self):
        """ To align with the image: Img[i, j] -> [x[i, j], y[i, j]]. Use top-left corner to represent the pixel.

            New: use the center of the pixel to represent the pixel.
        """
        ps = self.ps
        x, y = torch.meshgrid(
            torch.linspace(-0.5 * self.phy_size[0] + 0.5 * ps, 0.5 * self.phy_size[1] - 0.5 * ps, self.res[0]),
            torch.linspace(0.5 * self.phy_size[1] - 0.5 * ps, -0.5 * self.phy_size[0] + 0.5 * ps, self.res[1]), 
            indexing='xy')
        return x, y


    def gen_freq_grid(self):
        x, y = self.gen_xy_grid()
        fx = x / (self.ps * self.phy_size[0])
        fy = y / (self.ps * self.phy_size[1])
        return fx, fy


    def show(self, data='irr', save_name=None):
        """ TODO: use x, y coordinates

        Args:
            data (str, optional): _description_. Defaults to 'irr'.

        Raises:
            Exception: _description_
            Exception: _description_
        """
        if data == 'irr':
            value = self.u.detach().abs()**2
            cmap = 'gray'
        elif data == 'amp':
            value = self.u.detach().abs()
            cmap = 'gray'
        elif data == 'phi' or data == 'phase':
            value = torch.angle(self.u).detach()
            cmap = 'hsv'
        elif data == 'real':
            value = self.u.real.detach()
            cmap = 'gray'
        elif data == 'imag':
            value = self.u.imag.detach()
            cmap = 'gray'
        else:
            raise Exception('Unimplemented visualization.')

        if len(self.u.shape) == 2:
            if save_name is not None:
                save_image(value, save_name, normalize=True)
            else:
                value = value.cpu().numpy()
                plt.imshow(value, cmap=cmap, extent=[-self.phy_size[0]/2, self.phy_size[0]/2, -self.phy_size[1]/2, self.phy_size[1]/2])
        
        elif len(self.u.shape) == 4:
            B, C, H, W = self.u.shape
            if B == 1:
                if save_name is not None:
                    save_image(value, save_name, normalize=True)
                else:
                    value = value.cpu().numpy()
                    plt.imshow(value[0, 0, :, :], cmap=cmap, extent=[-self.phy_size[0]/2, self.phy_size[0]/2, -self.phy_size[1]/2, self.phy_size[1]/2])
            else:
                if save_name is not None:
                    plt.savefig(save_name)
                else:
                    value = value.cpu().numpy()
                    fig, axs = plt.subplots(1, B)
                    for i in range(B):
                        axs[i].imshow(value[i, 0, :, :], cmap=cmap, extent=[-self.phy_size[0]/2, self.phy_size[0]/2, -self.phy_size[1]/2, self.phy_size[1]/2])
                    fig.show()
        else:
            raise Exception('Unsupported complex field shape.')
            

    def Nyquist_zmin(self):
        """ Compute Nyquist zmin, suppose the second plane has the same side length with the original plane.
        """
        wvln = self.wvln * 1e-3 # [um] to [mm]
        zmin = np.sqrt((4 * self.ps**2 / wvln**2 - 1) * (self.phy_size[0]/2 + self.phy_size[0]/2)**2)
        return zmin


    # =============================================
    # 
    # =============================================

    def pad(self, Hpad, Wpad):
        """ Pad the input field by (Hpad, Hpad, Wpad, Wpad). 
            This step will also expand the field.

            NOTE: Can only pad plane field.

        Args:
            Hpad (_type_): _description_
            Wpad (_type_): _description_
        """
        # Pad real and imag
        # real = self.u.real
        # imag = self.u.imag
        # if len(real.shape) == 4:
        #     real = F.pad(real, (Hpad,Hpad,Wpad,Wpad), mode='constant', value=0)
        #     imag = F.pad(imag, (Hpad,Hpad,Wpad,Wpad), mode='constant', value=0)
        # elif len(real.shape) == 2:
        #     real = F.pad(real.unsqueeze(0).unsqueeze(0), (Hpad,Hpad,Wpad,Wpad), mode='constant', value=0).squeeze(0).squeeze(0)
        #     imag = F.pad(imag.unsqueeze(0).unsqueeze(0), (Hpad,Hpad,Wpad,Wpad), mode='constant', value=0).squeeze(0).squeeze(0)
        # u = real + 1j * imag
        # self.u = u

        device = self.device

        # Pad directly
        self.u = nnf.pad(self.u, (Hpad,Hpad,Wpad,Wpad), mode='constant', value=0)
        
        Horg, Worg = self.res
        self.res = [Horg + 2 * Hpad, Worg + 2 * Wpad]
        self.phy_size = [self.phy_size[0] * self.res[0] / Horg, self.phy_size[1] * self.res[1] / Worg]
        self.x, self.y = self.gen_xy_grid()
        z = self.z[0, 0]
        self.z = nnf.pad(self.z, (Hpad,Hpad,Wpad,Wpad), mode='constant', value=z)


    def clear_edge(self, Hpad, Wpad):
        """ Clear diffraction pattern in the edge(usually padding regions).

        Args:
            Hpad (_type_): _description_
            Wpad (_type_): _description_
        """
        if len(self.u.shape) == 2:
            self.u[:Hpad,:] = 0
            self.u[-Hpad:,:] = 0
            self.u[:,:Wpad] = 0
            self.u[:,-Wpad:] = 0
        elif len(self.u.shape) == 4:
            self.u[:,:,:Hpad,:] = 0
            self.u[:,:,-Hpad:,:] = 0
            self.u[:,:,:,:Wpad] = 0
            self.u[:,:,:,-Wpad:] = 0


    def get_valid(self):
        Hpad = int((self.phy_size[0] - self.valid_phy_size[0]) / self.phy_size[0] / 2 * self.res[0])
        Wpad = int((self.phy_size[1] - self.valid_phy_size[1]) / self.phy_size[1] / 2 * self.res[1])
        valid_u = self.u[:, :, Hpad:-Hpad, Wpad:-Wpad]
        return valid_u


    def scale_amp(self, scale):
        """_summary_

        Args:
            scale (_type_): _description_
        """
        raise Exception('Useless.')
        amp = self.u.abs()
        phi = torch.angle(self.u)
        self.u = scale * amp + 1j * phi 
        

# ===================================
# Diffraction functions
# ===================================

# def RayleighSommerfeldIntegral_old(field, z, n=1, Nmax=16*16, x=None, y=None):
#     """ Brute-force discrete Rayleigh-Sommerfeld-diffraction integration. This function should not be removed as it sometimes serves as the baseline and groud-truth to compare with.

#         Planar wave propogates to a near plane with same size.

#         There are two cases we should consider:
#         1): Input field has a very high resoultion, then the memory can be not enough.
#         2): Propagation z is below the minimum distance z0 required by Nyquist sampling criterion.

#     Args:
#         u (_type_): _description_
#         wvln (_type_): _description_
#         Nmax: Maximum integration points
#     """
#     H, W = field.res
#     wvln = field.wvln * 1e-3    # convert [um] to [mm]

#     # Input plane reshape tp [H*W]. x0f means "x0 flat"
#     x0f, y0f, z0f = field.x.reshape(-1), field.y.reshape(-1), field.z.reshape(-1)
#     u0f = field.u.reshape(-1)
    
#     # Output plane
#     x1 = x if x is not None else field.x
#     y1 = y if y is not None else field.y
#     z1 = z if torch.is_tensor(z) else torch.full_like(x1, z) 
#     # FIXME: we are using complex128
#     u1 = torch.zeros_like(x1, device=u0f.device, dtype=torch.complex128)
    
#     # Patch integration
#     Iter = int(np.ceil(H * W / Nmax))
#     for i in tqdm(range(Iter)):
#         lower = i * Nmax
#         upper = min((i + 1) * Nmax, H*W)
#         Npoint = upper - lower
#         x0sub, y0sub, z0sub, u0sub = x0f[lower:upper], y0f[lower:upper], z0f[lower:upper], u0f[lower:upper]
        
#         x1p = x1.unsqueeze(-1).repeat(1,1,Npoint)
#         y1p = y1.unsqueeze(-1).repeat(1,1,Npoint)
#         z1p = z1.unsqueeze(-1).repeat(1,1,Npoint)

#         r2 = (x1p-x0sub)**2 + (y1p-y0sub)**2 + (z1p-z0sub)**2
#         r = torch.sqrt(r2)
#         obliq = (z1p - z0sub) / r
#         # ==> Method 1: Classical RS integral
#         # u1 = 1 / (1j * wvln) * torch.sum(u0f * obliq / r * torch.exp(1j * torch.fmod(n * field.k * r, 2 * np.pi)) , -1)
#         u1sub = 1 / (1j * wvln) * torch.sum(u0sub * obliq / r * torch.exp(1j * torch.fmod(n * field.k * r, 2 * np.pi)) , -1)

#         # ==> Method 2: New form with 1/r^3
#         # u1 = torch.sum(u0f * z1p * torch.exp(1j * field.k * r) / (2 * np.pi * r**3) * (1 - 1j * field.k * r), -1)

#         # ==> Method 3: Eq.(3.1) in " Calculation of the Rayleigh..."
#         # u1 = - torch.sum(u0f * 2 * z1p / r * (1j * field.k - 1/r) * torch.exp(1j * field.k * r) / (4 * np.pi * r), -1) 

#         u1 += u1sub

#     pupil_func = 1
#     u1 = u1 / (H * W) * pupil_func
    
#     # TODO: this step we can only receive grid data
#     # FIXME: should optimize these lines
#     field1 = ComplexWave(u=u1, wvln=field.wvln, phy_size=[2*x1.abs().max().item(), 2*y1.abs().max().item()], device=field.device)
#     field1.x = x1
#     field1.y = y1
#     field1.z = z1
#     # plt.imshow(u1.cpu().abs()**2, cmap='gray')

#     return field1



def RayleighSommerfeldIntegral(u1, x1, y1, z, wvln, x2=None, y2=None, n=1., memory_saving=False):
    """ Brute-force discrete Rayleigh-Sommerfeld-diffraction integration. This function should not be removed as it sometimes serves as the baseline and groud-truth to compare with.

        Ref: https://diffractio.readthedocs.io/en/latest/source/tutorial/algorithms/RS.html

        Goodman: Introduction to Fourier Optics, 2rd edition, page 50, Eq.(3.43)

        Planar wave propogates to a near plane with same size.

        There are two cases we should consider:
        1): Input field has a very high resoultion, then the memory can be not enough.
        2): Propagation z is below the minimum distance z0 required by Nyquist sampling criterion.
        
        TODO: z can be a tensor or a float

        FIXME: this function consumes super large memory when constructing x1, x2, y1, y2. But the speed is super fast.

        TODO: can use float16 and complex32

    Args:
        u (_type_): _description_
        wvln (_type_): _description_
        Nmax: Maximum integration points
    """
    # Parameters
    assert wvln > 0.1 and wvln < 1, 'wvln unit should be [um].'
    k = n * 2 * np.pi / (wvln * 1e-3) # distance unit [mm]
    if x2 is None:
        x2 = x1.clone()
    if y2 is None:
        y2 = y1.clone()

    # Nyquist sampling criterion
    max_dist = (x1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x2.shape[0], x2.shape[1]) - x2).abs().max()
    ps = (x1.max() - x1.min()) / x1.shape[-1]
    zmin = Nyquist_zmin(wvln=wvln, ps=ps.item(), max_dist=max_dist.item(), n=n)
    assert zmin < z, 'Propagation distance is too short.'
    
    # Rayleigh Sommerfeld diffraction integral
    if memory_saving:
        u2 = torch.zeross_like(u1)
        step_size = 4
        x1 = x1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, step_size, step_size)    # [H1, W1, step_size, step_size]
        y1 = y1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, step_size, step_size)    # [H1, W1, step_size, step_size]
        u1 = u1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, step_size, step_size)    # [H1, W1, step_size, step_size]
        for i in range(0, x2.shape[0], step_size):
            for j in range(0, x2.shape[1], step_size):
                # Patch
                x2_patch = x2[i:i+step_size, j:j+step_size]
                y2_patch = y2[i:i+step_size, j:j+step_size]
                r2 = (x2_patch - x1)**2 + (y2_patch - y1)**2 + z**2 # shape of [H1, W1, step_size, step_size]
                r = torch.sqrt(r2)
                obliq = np.abs(z) / r
                # =======>
                # u2_patch = (- 1j * k) * torch.sum(u1 * obliq / r * torch.exp(1j * torch.fmod(k * r, 2 * np.pi)), (0, 1))
                # =======>
                u2_patch =  torch.sum(u1 * (z / r) * (1 / r - 1j * k) / (2 * np.pi * r) * torch.exp(1j * torch.fmod(k * r, 2 * np.pi)), (0, 1))
                # =======>
                
                # Assign
                u2[i:i+step_size, j:j+step_size] = u2_patch
    else:
        # Broadcast
        x1 = x1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x2.shape[0], x2.shape[1])    # shape of [H1, W1, H2, W2]
        y1 = y1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, y2.shape[0], y2.shape[1])    # shape of [H1, W1, H2, W2] 
        u1 = u1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x2.shape[0], x2.shape[1])    # shape of [H1, W1, H2, W2]

        # Rayleigh-Sommerfeld diffraction integral
        r2 = (x2 - x1)**2 + (y2 - y1)**2 + z**2 # shape of [H1, W1, H2, W2]
        r = torch.sqrt(r2)
        obliq = np.abs(z) / r
        # =======>
        # u2 =  (- 1j * k) * torch.sum(u1 * obliq / r * torch.exp(1j * torch.fmod(k * r, 2 * np.pi)), (0, 1))
        # =======>
        u2 =  torch.sum(u1 * (z / r) * (1 / r - 1j * k) / (2 * np.pi * r) * torch.exp(1j * torch.fmod(k * r, 2 * np.pi)), (0, 1))
        # =======>

    return u2


# def RSDint(x0, y0, z0, u0, x1, y1, z1, wvln, n=1, Nmax=64*64):
#     """ Brute-force discrete Rayleigh-Sommerfeld-diffraction integration.

#         https://cloud.tencent.com/developer/news/407424

#         There are two cases we should consider:
#         1): Input field has a very high resoultion, then the memory can be not enough.
#         2): Propagation z is below the minimum distance z0 required by Nyquist sampling criterion.

#         All inputs should be tensor.
#     """
#     raise Exception('This function is deprecated.')
#     H, W = u0.shape
#     wvln = wvln * 1e-3  # convert [um] to [mm]
#     k = 2 * np.pi / wvln

#     # Input plane reshape tp [H*W]. x0f means "x0 flat"
#     x0f, y0f, z0f = x0.reshape(-1), y0.reshape(-1), z0.reshape(-1)
#     u0f = u0.reshape(-1)
    
#     # Output plane
#     u1 = torch.zeros_like(x1, device=u0f.device, dtype=torch.complex128)
    
#     # Patch integration
#     Iter = int(np.ceil(H * W / Nmax))
#     logging.info(f'Totally {Iter} iterations.')
#     # We iterate input field
#     for i in range(Iter):
#     # for i in tqdm(range(Iter)): # TODO: tqdm
#         lower = i * Nmax
#         upper = min((i + 1) * Nmax, H*W)
#         Npoint = upper - lower
#         x0sub, y0sub, z0sub, u0sub = x0f[lower:upper], y0f[lower:upper], z0f[lower:upper], u0f[lower:upper]
        
#         x1p = x1.unsqueeze(-1).repeat(1,1,Npoint)
#         y1p = y1.unsqueeze(-1).repeat(1,1,Npoint)
#         z1p = z1.unsqueeze(-1).repeat(1,1,Npoint)

#         r2 = (x1p-x0sub)**2 + (y1p-y0sub)**2 + (z1p-z0sub)**2
#         r = torch.sqrt(r2)
#         # obliq = (z1p - z0sub) / r 
#         # ==> Method 1: Classical RS integral
#         # u1sub = 1 / (1j * wvln) * torch.sum(u0sub * obliq / r * torch.exp(1j * torch.fmod(n * k * r, 2 * np.pi)) , -1)
#         u1sub = 1 / (1j * wvln) * torch.sum(u0sub * (z1p - z0sub) / r2 * torch.exp(1j * torch.fmod(n * k * r, 2 * np.pi)) , -1)

#         # ==> Method 2: New form with 1/r^3
#         # u1 = torch.sum(u0f * z1p * torch.exp(1j * field.k * r) / (2 * np.pi * r**3) * (1 - 1j * field.k * r), -1)

#         # ==> Method 3: Eq.(3.1) in " Calculation of the Rayleigh..."
#         # u1 = - torch.sum(u0f * 2 * z1p / r * (1j * field.k - 1/r) * torch.exp(1j * field.k * r) / (4 * np.pi * r), -1) 

#         u1 += u1sub

#         if (i+1) % 10000 == 0:
#             logging.info(f'Iter {i+1}.')

#     pupil_func = 1
#     u1 = u1 / (H * W) * pupil_func
#     return u1






def AngularSpectrumMethod(u, z, wvln, ps, n=1., padding=True, TF=True):
    """ Rayleigh-Sommerfield propagation with FFT.

    Considerations:
        1, sampling requirement
        2, paraxial approximation
        3, boundary effects

        https://blog.csdn.net/zhenpixiaoyang/article/details/111569495

        TODO: add Nyquist creteria in this function

        TODO: double check TF and IR method.
    
    Args:
        u: complex field, shape [H, W] or [B, C, H, W]
        wvln: wvln
        res: field resolution
        ps (float): pixel size
        z (float): propagation distance
    """
    if torch.is_tensor(z):
        z = z.item()

    # Reshape
    if len(u.shape) == 2:
        Horg, Worg = u.shape
    elif len(u.shape) == 4:
        B, C, Horg, Worg = u.shape
        if isinstance(z, torch.Tensor):
            z = z.unsqueeze(0).unsqueeze(0)
    
    # Padding 
    if padding:
        Wpad, Hpad = Worg//2, Horg//2
        Wimg, Himg = Worg + 2*Wpad, Horg + 2*Hpad
        u = F.pad(u, (Wpad, Wpad, Hpad, Hpad))
    else:
        Wimg, Himg = Worg, Horg

    # Propagation
    assert wvln > 0.1 and wvln < 1, 'wvln unit should be [um].'
    k = 2 * np.pi / (wvln * 1e-3)   # we use k in vaccum, k in [mm]-1
    # x, y = torch.meshgrid(
    #     torch.linspace(-0.5 * Wimg * ps, 0.5 * Himg * ps, Wimg+1, device=u.device)[:-1],
    #     torch.linspace(0.5 * Wimg * ps, -0.5 * Himg * ps, Himg+1, device=u.device)[:-1],
    #     indexing='xy'
    # )
    # fx, fy = torch.meshgrid(
    #     torch.linspace(-0.5/ps, 0.5/ps, Wimg+1, device=u.device)[:-1],
    #     torch.linspace(-0.5/ps, 0.5/ps, Himg+1, device=u.device)[:-1],
    #     indexing='xy'
    # )
    x, y = torch.meshgrid(
        torch.linspace(-0.5 * Wimg * ps, 0.5 * Himg * ps, Wimg, device=u.device),
        torch.linspace(0.5 * Wimg * ps, -0.5 * Himg * ps, Himg, device=u.device),
        indexing='xy'
    )
    fx, fy = torch.meshgrid(
        torch.linspace(-0.5/ps, 0.5/ps, Wimg, device=u.device),
        torch.linspace(0.5/ps, -0.5/ps, Himg, device=u.device),
        indexing='xy'
    )

    # Determine TF or IR
    if ps > wvln * np.abs(z) / (Wimg * ps):
        TF = True
    else:
        TF = False

    if TF: 
        if n == 1:
            square_root = torch.sqrt(1 - (wvln*1e-3)**2 * (fx**2 + fy**2))
            H = torch.exp(1j * k * z * square_root)
        else:
            square_root = torch.sqrt(n**2 - (wvln*1e-3)**2 * (fx**2 + fy**2))
            H = n * torch.exp(1j * k * z * square_root)
        
        H = fftshift(H)
    
    else:
        r2 = x**2 + y**2 + z**2
        r = torch.sqrt(r2)

        if n == 1:
            h = z / (1j * wvln * r2) * torch.exp(1j * k * r)
        else:
            h = z * n / (1j * wvln * r2) * torch.exp(1j * n * k * r)
        
        H = fft2(fftshift(h)) * ps**2
    

    # Fourier transformation
    # https://pytorch.org/docs/stable/generated/torch.fft.fftshift.html#torch.fft.fftshift
    u = ifftshift(ifft2(fft2(fftshift(u)) * H))

    # Remove padding
    if padding:
        u = u[..., Wpad:-Wpad, Hpad:-Hpad]

    del x, y, fx, fy
    return u


def FresnelDiffraction(u, z, wvln, ps, n=1., padding=True, TF=None):
    """ Fresnel propagation with FFT.

    Ref: Computational fourier optics : a MATLAB tutorial
         https://github.com/nkotsianas/fourier-propagation/blob/master/FTFP.m

    Args:
        u (_type_): _description_
        wvln (_type_): _description_
        ps (_type_): _description_
        z (_type_): _description_
        n (_type_, optional): _description_. Defaults to 1..
        padding (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    # padding 
    if padding:
        try:
            _, _, Worg, Horg = u.shape
        except:
            Horg, Worg = u.shape
        Wpad, Hpad = Worg//2, Horg//2
        Wimg, Himg = Worg + 2 * Wpad, Horg + 2 * Hpad
        u = F.pad(u, (Wpad, Wpad, Hpad, Hpad))
    else:
        _, _, Wimg, Himg = u.shape

    # compute H function
    assert wvln < 10, 'wvln should be in [um].'
    k = 2 * np.pi / wvln
    x, y = torch.meshgrid(
        torch.linspace(-0.5 * Wimg * ps, 0.5 * Himg * ps, Wimg+1, device=u.device)[:-1],
        torch.linspace(0.5 * Wimg * ps, -0.5 * Himg * ps, Himg+1, device=u.device)[:-1],
        indexing='xy'
    )
    fx, fy = torch.meshgrid(
        torch.linspace(-0.5/ps, 0.5/ps, Wimg+1, device=u.device)[:-1],
        torch.linspace(-0.5/ps, 0.5/ps, Himg+1, device=u.device)[:-1],
        indexing='xy'
    )
    # fx, fy = x/ps, y/ps

    # Determine TF or IR
    if TF is None:
        if ps > wvln * np.abs(z) / (Wimg * ps):
            TF = True
        else:
            TF = False
    # TF = True
    
    # Computational fourier optics. Chapter 5, section 5.1.
    if TF:
        # Correct, checked.
        if n == 1:
            # H = torch.exp(- 1j * torch.fmod(np.pi * wvln * z * (fx**2 + fy**2), 2 * np.pi))
            H = torch.exp(- 1j * np.pi * wvln * z * (fx**2 + fy**2))
        else:
            # H = np.sqrt(n) * torch.exp(- 1j * torch.fmod(np.pi * wvln * z * (fx**2 + fy**2) / n, 2 * np.pi))
            H = np.sqrt(n) * torch.exp(- 1j * np.pi * wvln * z * (fx**2 + fy**2) / n)

        H = fftshift(H)
    else:
        if n == 1:
            # h = 1 / (1j * wvln * z) * torch.exp(1j * torch.fmod(k / (2*z) * (x**2+y**2), 2 * np.pi))   
            h = 1 / (1j * wvln * z) * torch.exp(1j * k / (2*z) * (x**2+y**2)) 
        else:
            # h = n / (1j * wvln * z) * torch.exp(1j * torch.fmod(n * k / (2*z) * (x**2+y**2), 2 * np.pi))
            h = n / (1j * wvln * z) * torch.exp(1j * n * k / (2*z) * (x**2+y**2))
        
        H = fft2(fftshift(h)) * ps**2
    
    
    # Fourier transformation
    # https://pytorch.org/docs/stable/generated/torch.fft.fftshift.html#torch.fft.fftshift
    u = ifftshift(ifft2(fft2(fftshift(u)) * H))

    # remove padding
    if padding:
        u = u[...,Wpad:-Wpad,Hpad:-Hpad]

    return u


def FraunhoferDiffraction(u, z, wvln, ps, n=1., padding=True):
    """ Fraunhofer propagation. 
    """
    # padding 
    if padding:
        Worg, Horg = u.shape
        Wpad, Hpad = Worg//4, Horg//4
        Wimg, Himg = Worg + 2*Wpad, Horg + 2*Hpad
        u = F.pad(u, (Wpad, Wpad, Hpad, Hpad))
    else:
        Wimg, Himg = u.shape

    # side length
    L2 = wvln * z / ps
    ps2 = wvln * z / Wimg / ps
    x2, y2 = torch.meshgrid(
        torch.linspace(- L2 / 2, L2 / 2, Wimg+1, device=u.device)[:-1],
        torch.linspace(- L2 / 2, L2 / 2, Himg+1, device=u.device)[:-1],
        indexing='xy'
    ) 


    # Computational fourier optics. Chapter 5, section 5.5.
    # Shorter propagation will not affect final results.
    
    k = 2 * np.pi / wvln
    if n == 1:
        # c = 1 / (1j * wvln * z) * torch.exp(1j * torch.fmod(k / (2 * z) * (x2 ** 2 + y2 ** 2), 2 * np.pi))
        c = 1 / (1j * wvln * z) * torch.exp(1j * k / (2 * z) * (x2 ** 2 + y2 ** 2))
    else:
        # c = n / (1j * wvln * z) * torch.exp(1j * torch.fmod(n * k / (2 * z) * (x2 ** 2 + y2 ** 2), 2 * np.pi))
        c = n / (1j * wvln * z) * torch.exp(1j * n * k / (2 * z) * (x2 ** 2 + y2 ** 2))
    
    # FIXME: n != 1, how to write fft2
    u = c * ps ** 2 * ifftshift(fft2(fftshift(u)))

    # remove padding
    if padding:
        u = u[...,Wpad:-Wpad,Hpad:-Hpad]

    return u


def RSD_batch(x0, y0, z0, u0, x1, y1, z1, wvln, n=1, normal=None):
    """ Brute-force RS integral with bacth input.

    Args:
        x0 (_type_): _description_
        y0 (_type_): _description_
        z0 (_type_): _description_
        u0 (_type_): _description_
        x1 (_type_): _description_
        y1 (_type_): _description_
        z1 (_type_): _description_
        wvln (_type_): _description_
        n (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    raise Warning('This function should be checked when used.')
    wvln = wvln * 1e-3  # convert [um] to [mm]
    k = 2 * np.pi / wvln

    B, C, H0, W0 = u0.shape
    x0 = x0.unsqueeze(0).unsqueeze(0)
    y0 = y0.unsqueeze(0).unsqueeze(0) 
    z0 = z0.unsqueeze(0).unsqueeze(0)

    # Output plane
    H1, W1 = x1.shape
    u1 = torch.zeros((B, C, H1, W1), dtype=torch.complex128).to(u0.device)

    logging.info(f'Totally {H1 * W1} iterations.')

    for i in range(H1):
        for j in range(W1):
            x1_0 = x1[i, j]
            y1_0 = y1[i, j]
            z1_0 = z1[i, j]
            
            r2 = (x1_0 - x0)**2 + (y1_0 - y0)**2 + (z1_0 - z0)**2
            r = torch.sqrt(r2)

            phi = torch.fmod(n * k * r, 2 * np.pi)
            amp = u0 * (z1_0 - z0) / r2

            # ==> Classical solution
            # u1_0 = n / (1j * wvln) * torch.sum(torch.sum(amp * torch.exp(1j * phi) , -1), -1)

            # ==> More accurate solution
            if normal is not None:
                normal_0 = normal[i, j, :]

                jkr = 1j * n * k + 1 / r    # ik + 1/r term in "Non-approximated Rayleigh–Sommerfeld diffraction integral: advantages and disadvantages in the propagation of complex wave fields"

                oblique = (x1_0 - x0) * normal_0[0] + (y1_0 - y0) * normal_0[1] + (z1_0 - z0) * normal_0[2]
                amp = u0 * oblique / r2

                u1_0 = - 1 / (2 * np.pi) * torch.sum(torch.sum(amp * jkr * torch.exp(1j * phi) , -1), -1)
            
            else:
                u1_0 = n / (1j * wvln) * torch.sum(torch.sum(amp * torch.exp(1j * phi) , -1), -1)
            

            u1[:, :, i, j] = u1_0
            if (i * H1 + j) % 100000 == 0:
                logging.info(f'Finish {i * H1 + j} iterations.')

    u1 = u1 / (H0 * W0)
    return u1



# ===================================
# Utils
# ===================================

def Nyquist_criterion(L1, L2, wvln, z):
    """ Compute Nyquist sampling frequency.

    Args:
        L1 (_type_): Input field side length.
        L2 (_type_): Output field side length.
        wvln (_type_): wvln in [um].
        z (_type_): Propagation distance.
    """
    wvln = wvln * 1e-3  # convert [um] to [mm]
    ps1_max = wvln * np.sqrt((L1**2 + L2**2) + z**2) / 2 / (L1 + L2)
    input_field_res_min = L1 / ps1_max

    return


def Nyquist_zmin(wvln, ps, max_dist, n=1.):
    """ Nyquist sampling condition for Rayleigh Sommerfeld diffraction.

    Args:
        wvln: wvln in [um]
        ps: pixel size in [mm]
        max_len: maximum side distance between input and output field in [mm]
        n: refractive index
    """
    wvln = wvln * 1e-3  # [um] to [mm]
    zmin = np.sqrt((4 * ps**2 * n**2 / wvln**2 - 1)) * max_dist
    return zmin


def Nyquist_ps_max(wvln, z, halfL1, halfL2, n=1):
    wvln = wvln * 1e-3
    ps_max = np.sqrt(z**2 / (halfL1 + halfL2)**2 + 1) * wvln / 2 / n
    return ps_max


def Nyquist_N_min(wvln, z, halfL1, halfL2, n=1):
    wvln = wvln * 1e-3
    f_max = n * (halfL1 + halfL2) / wvln / np.sqrt((halfL1 + halfL2) **2 + z**2)
    N_min = 2 * f_max * 2 * halfL1
    N_min *= 1.05   # slightly increase sampling density
    return int(np.ceil(N_min))


def FFT_ps(wvln, z, L):
    """ If we use TF function, we need "ps >= wvln * z / L" to oversample H.
        If we use IF function, we need "ps <= wvln * z / L" to oversample h.

    Args:
        wvln (_type_): _description_
        z (float, Tensor): _description_
        L (_type_): _description_

    Returns:
        _type_: _description_
    """
    if torch.is_tensor(z):
        z = torch.mean(z).item() 

    wvln = wvln * 1e-3
    ps = wvln * np.abs(z) / L
    return ps
