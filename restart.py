import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('restart')
log.setLevel(logging.DEBUG)
log.info("restart process started, it will take few mins \n all models wil rebuild \n and newly added authors file will be included in training ")

log.info("*********** Restarting Compression Model ***************")
import Backend.compression_Model.compression_model
log.info("*********** compression_model: Restart completed ***************")

log.info("*********** Restarting ngram Model ***************")
import Backend.ngram_Model.ngram_model
log.info("*********** ngram_model: Restart completed  ***************")

log.info("*********** Restarting Lexical Model ***************")
import Backend.lexical_Model.lexical_feature_extracting_v2
import Backend.lexical_Model.ConvertInput
import Backend.lexical_Model.lexical_model
log.info("*********** lexical_model: Restart completed  ***************")

log.info("*********** Restarting polysemy Model ***************")
import Backend.polysemy_Model.Extract_polysemy
import Backend.polysemy_Model.poly_model
log.info("*********** polysemy_model: Restart completed  ***************")