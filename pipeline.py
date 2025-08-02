from sklearn.base import BaseEstimator, TransformerMixin
from gudhi.representations.vector_methods import BettiCurve
from gudhi.point_cloud.timedelay import TimeDelayEmbedding
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from gudhi.sklearn.rips_persistence import RipsPersistence
from gudhi.representations import DiagramSelector, Landscape


class BatchModule(BaseEstimator, TransformerMixin):
    def __init__(self, module, *args, **kwargs):
        self.module = module
        self.args = args
        self.kwargs = kwargs
        self.models_ = []

    def fit(self, X, y=None):
        self.models_ = []
        for cloud in X:
            model = self.module(
                *self.args, **self.kwargs
            )
            model.fit(cloud)
            self.models_.append(model)
        return self

    def transform(self, X):
        return [model.transform(cloud) for model, cloud in zip(self.models_, X)]

embedding_dimension = 200
embedding_time_delay = 1
stride = 5
pipeline_embedder = Pipeline(
    [
        ("embedder", TimeDelayEmbedding(dim=embedding_dimension, delay=embedding_time_delay, skip=stride)),
        ("PCA", BatchModule(PCA, n_components=3)),
    ]
)

pipeline_complex = Pipeline(
    [
        ("rips_pers", RipsPersistence(homology_dimensions=(0,1), n_jobs=-1, homology_coeff_field=2, num_collapses=0)),
        ("finite_diags", BatchModule(DiagramSelector, use=True, point_type="finite")),
        ("betti_cruve", BatchModule(BettiCurve, resolution=100)),
    ]
)

full_pipe = Pipeline(
    [
        ("embed", pipeline_embedder), 
        ("complex", pipeline_complex),
    ]
)