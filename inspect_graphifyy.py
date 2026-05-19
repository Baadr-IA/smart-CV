import graphifyy
import inspect
import pkgutil
print('graphifyy', graphifyy)
print('version', getattr(graphifyy, '__version__', 'unknown'))
print('doc', graphifyy.__doc__[:400] if graphifyy.__doc__ else 'no doc')
print('submodules:')
for m in pkgutil.iter_modules(graphifyy.__path__):
    print(' -', m.name)
print('attrs:')
for name in sorted(n for n in dir(graphifyy) if not n.startswith('_')):
    print(name)