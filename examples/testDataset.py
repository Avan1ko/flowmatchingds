import scipy.io
mat = scipy.io.loadmat(r'C:\Users\ericd\My Drive\5th Year\Spring\MEAM6230\flowmatchingds\examples\Angle.mat')
print(mat.keys()) 
# Look for 'demos' or 'data'. 
# Then check the type of the first demo:
first_item = mat['demos'][0, 0]
print(type(first_item))
print(first_item.dtype)