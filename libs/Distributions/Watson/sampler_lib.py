
def InvTransSamp(data, n_bins, n_samples):
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    r = np.random.rand(n_samples)
    return inv_cdf(r)


from scipy.special import hyp1f1
from scipy.special import gamma
import numpy as np
import scipy.interpolate as interpolate
import utilities_lib as ul

def InvTransSampGrid(pdf_Values, pdf_Grid, Nsam):
    # To this function we give a grid of the pdf and it outputs variables
    # In this case cdf_Values are the values of the cdf and cdf_Grid the variables
    # pdf_Grid = [X1, X2...]
    pdf_Grid = ul.fnp(pdf_Grid)
    pdf_Values = ul.fnp(pdf_Values)
    
    print (pdf_Grid.shape)
    print (pdf_Grid[[-1],:].shape)
    pdf_Grid = np.concatenate((pdf_Grid, pdf_Grid[[-1],:]), axis = 0)
    
    cum_values = np.zeros(pdf_Grid.shape)
    print (cum_values.shape)
    
    ## Get the dimensions matrix
    
    HyperVolume = np.diff(pdf_Grid, axis = 0)
    print (HyperVolume.shape)
    
    NormedCDF = np.cumsum(pdf_Values*HyperVolume)
    NormedCDF = ul.fnp(NormedCDF)
    print (NormedCDF.shape)
    
    cum_values[1:,:] = NormedCDF
    print ("----")
    print ([pdf_Grid[:-1,:].shape, NormedCDF.shape])
        
    inv_cdf = interpolate.interp1d(NormedCDF.flatten(), pdf_Grid[:-1,:].flatten())
    r = np.random.rand(Nsam)
    return inv_cdf(r)

#
#
#def  vmfrnd(m, n, kappa, mu):
#    #RANDVONMISESFISHERM Random number generation from von Mises Fisher
#    #distribution.
#    #X = randvonMisesFisherm(m, n, kappa) returns n samples of random unit
#    #directions in m dimensional space, with concentration parameter kappa,
#    #and the direction parameter mu = e_m
#    #X = randvonMisesFisherm(m, n, kappa, mu) with direction parameter mu
#    #(m-dimensional column unit vector)
#    
#    if m < 2:
#       print('Dimension m must be > 2. Dimension set to 2');
#       m = 2
#    
#    if kappa < 0:
#       disp('appa must be >= 0, Set kappa to be 0');
#       kappa = 0
#
#    
#    #the following algorithm is following the modified Ulrich's algorithm
#    # discussed by Andrew T.A. Wood in "SIMULATION OF THE VON MISES FISHER
#    # DISTRIBUTION", COMMUN. STATIST 23(1), 1994.
#    
#    # step 0 : initialize
#    b = (-2*kappa + np.sqrt(4*kappa**2 + (m-1)**2))/(m-1);
#    x0 = (1-b)/(1+b);
#    c = kappa*x0 + (m-1)*log(1-x0**2);
#    
#    # step 1 & step 2
#    nnow = n; w = [];
#    
#    while(True):
#       ntrial = max(round(nnow*1.2),nnow+10) ;
#       Z = betarnd((m-1)/2,(m-1)/2,ntrial,1);
#       U = np.random.randn(ntrial,1);
#       W = (1-(1+b)*Z)/(1-(1-b)*Z);
#       
#       indicator = kappa*W + (m-1)*log(1-x0*W) - c >= log(U);
#       if sum(indicator) >= nnow
#          w1 = W(indicator);
#          w = [w ;w1(1:nnow)];
#          break;
#       else
#          w = [w ; W(indicator)];
#          nnow = nnow-sum(indicator);
#          %cnt = cnt+1;disp(['retrial' num2str(cnt) '.' num2str(sum(indicator))]);
#       end
#    end
#    
#    % step 3
#    V = UNIFORMdirections(m-1,n);
#    X = [repmat(sqrt(1-w'.^2),m-1,1).*V ;w'];
#    
#    if muflag
#       mu = mu / norm(mu);
#       X = rotMat(mu)'*X;
#    end
#    end
#    
#    
#    function V = UNIFORMdirections(m,n)
#    % generate n uniformly distributed m dim'l random directions
#    % Using the logic: "directions of Normal distribution are uniform on sphere"
#    
#    V = zeros(m,n);
#    nr = randn(m,n); %Normal random
#    for i=1:n
#       while 1
#          ni=nr(:,i)'*nr(:,i); % length of ith vector
#          % exclude too small values to avoid numerical discretization
#          if ni<1e-10
#             % so repeat random generation
#             nr(:,i)=randn(m,1);
#          else
#             V(:,i)=nr(:,i)/sqrt(ni);
#             break;
#          end
#       end
#    end
#
#
#
#function rot = rotMat(b,a,alpha)
#% ROTMAT returns a rotation matrix that rotates unit vector b to a
#%
#%   rot = rotMat(b) returns a d x d rotation matrix that rotate
#%   unit vector b to the north pole (0,0,...,0,1)
#%
#%   rot = rotMat(b,a ) returns a d x d rotation matrix that rotate
#%   unit vector b to a
#%
#%   rot = rotMat(b,a,alpha) returns a d x d rotation matrix that rotate
#%   unit vector b towards a by alpha (in radian)
#%
#%    See also .
#
#% Last updated Nov 7, 2009
#% Sungkyu Jung
#
#
#[s1 s2]=size(b);
#d = max(s1,s2);
#b= b/norm(b);
#if min(s1,s2) ~= 1 || nargin==0 , help rotMat, return, end
#
#if s1<=s2;    b = b'; end
#
#if nargin == 1;
#   a = [zeros(d-1,1); 1];
#   alpha = acos(a'*b);
#end
#
#if nargin == 2;
#   alpha = acos(a'*b);
#end
#if abs(a'*b - 1) < 1e-15; rot = eye(d); return, end
#if abs(a'*b + 1) < 1e-15; rot = -eye(d); return, end
#
#c = b - a * (a'*b); c = c / norm(c);
#A = a*c' - c*a' ;
#
#rot = eye(d) + sin(alpha)*A + (cos(alpha) - 1)*(a*a' +c*c');
#end