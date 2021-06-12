from sklearn.base import BaseEstimator, TransformerMixin


class StacsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, filter_characteristics='none', drop_id_bounding=False):
        self.filter_characteristics = filter_characteristics
        self.drop_id_bounding = drop_id_bounding

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Drop ID and bounding box features as specified
        if self.drop_id_bounding:
            keep = []
        else:
            keep = ['image', 'x', 'y', 'w', 'h']

        # Reduce features to characteristics extracted from images as specified
        to_drop = []
        if self.filter_characteristics == 'hog' or self.filter_characteristics == 'bimp':
            characteristic = self.filter_characteristics
            for col in X.columns:
                if col not in keep and characteristic not in col:
                    to_drop.append(col)
        elif self.filter_characteristics == 'cielab':
            for col in X.columns:
                if col not in keep and \
                        (('lightness' not in col) and ('redgreen' not in col) and ('blueyellow' not in col)):
                    to_drop.append(col)
        X = X.drop(columns=to_drop)

        return X
