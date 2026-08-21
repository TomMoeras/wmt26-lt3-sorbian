# REQUIRED job-local shim (see environment/ENVIRONMENT.md "the kernels shim").
# kernels==0.16.0 is binary-incompatible with transformers==5.9.0: importing a model class
# builds a LayerRepository that raises "Either a revision or a version must be specified."
# Making `import kernels` raise a *caught* ImportError forces transformers to fall back to its
# pure-torch path. This directory must be on PYTHONPATH for every generate/train run; the run
# scripts source shims/activate.sh, which does exactly that.
raise ImportError("kernels hub package disabled job-locally "
                  "(transformers 5.9.0 x kernels 0.16.0 LayerRepository incompat)")
