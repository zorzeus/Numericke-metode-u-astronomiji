import numpy as np
#------------------------------------------------------------------------------
def spline_coefs(x,y):
# =============================================================================
# calculates natural cubic spline coefficient for the nodes x,y
# input: vectors of the nodes coordinates x, y
# output: vectors of the coefficients for the piecewise polinomial of the form
# a+b*(x-x[i])+c*(x-x[i])**2+d*(x-x[i])**3
# where i is the index of the node whose x-coordinate is the largest amog those which are smaller than x
#(first on the left from x)
# remark:
# parameter is t=x-x_i
# works with non-equidistant data
# =============================================================================
    a=y[:-1]
    b=np.zeros(len(a), dtype='float')
    d=np.zeros(len(a), dtype='float')
    h=np.zeros(len(x)-1, dtype='float')
    for i in range(0,len(x)-1):
        h[i]=x[i+1]-x[i]
        
    A=np.zeros([len(x), len(x)], dtype='float')
    v=np.zeros(len(x))
    
    for i in range(1,len(A)-1):
    
        A[i][i]=2*(h[i-1]+h[i])
        A[i][i-1]=h[i-1]
        A[i][i+1]=h[i]
        
        v[i]=3*((y[i+1]-y[i])/h[i]-(y[i]-y[i-1])/h[i-1])
    
    A[0,0]=1; A[-1][-1]=1;
    
                  
    c = np.linalg.solve(A,v)
    
    for i in range(len(a)):
        b[i]=(y[i+1]-y[i])/h[i]-h[i]/3*(2*c[i]+c[i+1])
        d[i]=(c[i+1]-c[i])/3/h[i]
        
    c=c[:-1]
    
    return a,b,c,d

#------------------------------------------------------------------------------
def spline_interp(x,y,x0):
# =============================================================================
# calculates interpolated values by natural cubic spline interpolation
# input: x,y - nodes vectors
#        x0 - vector of arguments for which interpolated values are calculated
# output: y0 - interpolated values
# =============================================================================
    a,b,c,d=spline_coefs(x,y)
    
    y0=np.zeros(len(x0))
    for i in range(len(y0)):

        ind=np.argwhere(x<=x0[i])[-1]
        
        if ind==len(a): # za posljednji segment
            ind=ind-1

        y0[i]=a[ind]+b[ind]*(x0[i]-x[ind]) \
        +c[ind]*(x0[i]-x[ind])**2+d[ind]*(x0[i]-x[ind])**3

    return y0
#------------------------------------------------------------------------------
def spline_der(x,y,x0,n):
# =============================================================================
# calculates derivatives by using natural cubic spline interpolation
# input: x,y - nodes vectors
#        n - order of differentiation (1 or 2)
#        x0 - vector of arguments for which derivatives are calculated
# output: y0 - dy/dx at x0
# =============================================================================    
    a,b,c,d=spline_coefs(x,y)
    der=np.zeros(len(x0))
    if n==1:
        for i in range(len(der)):
            ind=np.argwhere(x<=x0[i])[-1]
            der[i]=b[ind]+2*c[ind]*(x0[i]-x[ind])+3*d[ind]*(x0[i]-x[ind])**2
        
    elif n==2:
        for i in range(len(der)):
            ind=np.argwhere(x<=x0[i])[-1]
            der[i]=2*c[ind]+6*d[ind]*(x0[i]-x[ind])
        
    return der
#------------------------------------------------------------------------------

def spline_integrate(x,y,x1,x2):
# =============================================================================
# calculates area under the natural spline with nodes x,y from x1 to x2
# input: x,y - nodes vectors
#        x1,x2 - limits of integration
# output: area
# =============================================================================
    i0=np.argwhere(x<=x1)[-1][0]
    i1=np.argwhere(x<=x2)[-1][0]
    
    if i1==len(x)-1:
        i1-=1
    
    a,b,c,d=spline_coefs(x,y)
    
    if i1==i0: # x1 i x2 se nalaze na istom segmentu
        tp=x1-x[i0]
        tk=x2-x[i0]
        A=a[i0]*(tk-tp)+b[i0]/2*(tk**2-tp**2)+c[i0]/3*(tk**3-tp**3)+d[i0]/4*(tk**4-tp**4)
    elif i1-i0==1: # x1 i x2 se nalaze na susjednim intervalima
      
        # prvi segment
        tp=x1-x[i0]
        tk=x[i1]-x1
        A=a[i0]*(tk-tp)+b[i0]/2*(tk**2-tp**2)+c[i0]/3*(tk**3-tp**3)+d[i0]/4*(tk**4-tp**4)
      
        # drugi segment
        tp=0
      
        tk=x2-x[i1]
        A+=a[i1]*(tk-tp)+b[i1]/2*(tk**2-tp**2)+c[i1]/3*(tk**3-tp**3)+d[i1]/4*(tk**4-tp**4)
  
    else: # između x1 i x2 postoji vise cijelih segmenata
      
        # prvi segment
        tp=x1-x[i0]
        tk=x[i0+1]-x1
        A=a[i0]*(tk-tp)+b[i0]/2*(tk**2-tp**2)+c[i0]/3*(tk**3-tp**3)+d[i0]/4*(tk**4-tp**4)
        # poslednji segment
      
        tp=0
        tk=x2-x[i1]
        A+=a[i1]*(tk-tp)+b[i1]/2*(tk**2-tp**2)+c[i1]/3*(tk**3-tp**3)+d[i1]/4*(tk**4-tp**4)
        
        # segmenti između
        for i in range(i0+1,i1):
            tp=0
            tk=x[i+1]-x[i]
            A+=a[i]*(tk-tp)+b[i]/2*(tk**2-tp**2)+c[i]/3*(tk**3-tp**3)+d[i]/4*(tk**4-tp**4)
    
    return A
#------------------------------------------------------------------------------
def inverse_interp(x,y,y0):
# =============================================================================
# performes inverse natural cubic spline interpolation
# input: x,y - nodes vectors
#        y0 - vector of function values for which arguments are calculated
# output: x0 - vector of arguments for which function has values y0
# =============================================================================    
    a,b,c,d=spline_coefs(x,y)
    
    x0=[]
    for j in range(len(y0)):
        xx=[]
        for i in range(len(a)):
            t=np.roots([d[i],c[i],b[i],a[i]-y0[j]]) # traži nule polinoma
            t=t[(np.argwhere(np.imag(t)==0)).flatten()] # uzima samo realne nule
            t=t[(np.argwhere(t>0)).flatten()] # uzima samo nule koje su vece od 0
            t=t[(np.argwhere(t<x[i+1]-x[i])).flatten()] # uzima samo nule koje su manje od x[i+1]-x[i])
            if len(t)>0: 
                for k in range(len(t)):
                    xx.append(t[k].real+x[i])
        x0.append(xx)     
    return x0[0]

#------------------------------------------------------------------------------

def extrema(x,y,minmax):
# =============================================================================
# finds all local minimums or maximums of the function
# input: x,y - nodes vectors
# if minmax='min' the output are local minimums:
# if minmax='max' the output are local maximums

# output: x0 - vector of arguments for which function has extreme values (min or max)
#         y0 - values of the function at extreme points (min or max)
# =============================================================================      
    a,b,c,d=spline_coefs(x,y)
    x0=np.array([]) # mjesta na kojima je prvi izvod jednak nuli
    d2=np.array([]) # vrijednost drugog izvoda
    
    for i in range(len(a)):
        t=np.roots([3*d[i],2*c[i],b[i]])
        t=t[(np.argwhere(np.imag(t)==0)).flatten()]
        t=t[(np.argwhere(t>0)).flatten()]
        t=t[(np.argwhere(t<x[i+1]-x[i])).flatten()]
        x0=np.append(x0,np.real(t)+x[i]) 
    x0=np.sort(x0)

    d2=np.zeros(len(x0))
    for i in range(len(x0)):
        ind=np.argwhere(x<=x0[i])[-1]
        d2[i]=2*c[ind]+6*d[ind]*(x0[i]-x[ind])

    if minmax=='min':
        x0=x0[np.argwhere(d2>0)]

    elif minmax=='max':
        x0=x0[np.argwhere(d2<0)]
    
    y0=spline_interp(x,y,x0)  
    return x0,y0
