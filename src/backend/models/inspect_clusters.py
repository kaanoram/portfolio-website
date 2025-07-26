import joblib 
import sys
import os
import logging 
from utils import apply_weights

# Add the parent directory (backend) to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the 'models' directory
parent_dir = os.path.dirname(script_dir)  # Gets the 'backend' directory
sys.path.append(parent_dir)

logging.basicConfig(level=logging.INFO, format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

file_path = "enhanced_clustering_model.joblib"
try:
    clustering_data = joblib.load(file_path)
    logger.info("Successfully loaded clustering data from joblib file")
except Exception as e:
    logger.error(f"Error loading joblib file: {str(e)}")
    raise

cluster_profiles = clustering_data.get("cluster_profiles", None)
if cluster_profiles:
    logger.info("Cluster Profiles:")
    for cluster_name, profile in cluster_profiles.items():
        logger.info(f"\n{cluster_name}:")
        logger.info(f"  Size: {profile['size']} ({profile['size_percentage']:.1f}%)")
        logger.info(f"  Monetary Metrics:")
        for key, value in profile['monetary_metrics'].items():
            logger.info(f"  {key}: {value}")
        logger.info("   Behavioral Metrics:")
        for key, value in profile['behavioral_metrics'].items():
            logger.info(f"  {key}: {value}")
        logger.info("   Temporal Patterns:")
        for key, value in profile['temporal_patterns'].items():
            logger.info(f"  {key}: {value}")
else:
    logger.warning("No cluster_profiles found in the joblib file")

