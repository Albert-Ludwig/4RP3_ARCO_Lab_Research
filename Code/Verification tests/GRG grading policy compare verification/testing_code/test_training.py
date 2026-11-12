"""Verify physical-unit posterior matches the existing trained GPyTorch model."""
import importlib.util
import sys
import numpy as np
from nav2_policy.prepare import REPO,ROOT
from nav2_policy.gp import GaussianProcess
from nav2_policy.storage import save_json

source=REPO/'Code/GP3D_baseline_test/10.16_testing_code/3D_time&spatial temp field rebuild & prediction/gp_model.py'
spec=importlib.util.spec_from_file_location('reused_training_validation',source)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
rng=np.random.default_rng(7);x=rng.uniform([0,0,-.2],[5,5,-.01],size=(24,3))
y=20.+np.sin(x[:,0])+.3*x[:,1]+x[:,2]
settings=dict(threads=1,iterations=12,learning_rate=.05,initial_lengthscale_scaled=1.,initial_noise_variance_scaled=.05,validation_every=4)
fit=module.fit_gp(x,y,None,None,settings,seed=0,select_by_validation=False)
hp=fit.hyperparameters();hp['lengthscales_m_m_s']=[*hp['lengthscales_m_m_h'][:2],hp['lengthscales_m_m_h'][2]*3600]
physical=x.copy();physical[:,2]*=3600
gp=GaussianProcess(physical,y,hp)
q=rng.uniform([0,0,-.1],[5,5,.05],size=(13,3));original=fit.predict(q)
qs=q.copy();qs[:,2]*=3600;mean,variance=gp.posterior(qs)
np.testing.assert_allclose(mean,original['mean'],atol=1e-7,rtol=1e-7)
np.testing.assert_allclose(variance,original['latent_var'],atol=1e-7,rtol=1e-7)
result=dict(passed=True,scope='synthetic numerical equivalence, not field-model validation',
    maximum_mean_error_c=float(np.max(abs(mean-original['mean']))),
    maximum_variance_error_c2=float(np.max(abs(variance-original['latent_var']))),
    reused_source=str(source),time_conversion='old hours -> new seconds, lengthscale multiplied by 3600')
save_json(ROOT/'result/training_equivalence.json',result);print(result)
