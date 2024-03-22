from gluefactory_nonfree import superpoint
from gluefactory.models.matchers import lightglue
from gluefactory.models.matchers.lightglue import LightGlue

from gluefactory.models.matchers import lightglue_pe4D
from gluefactory.models.matchers.lightglue_pe4D import LightGlue4D

model = lightglue.LightGlue(conf=LightGlue.default_conf)
model4D = lightglue_pe4D.LightGlue4D(conf=LightGlue4D.default_conf)