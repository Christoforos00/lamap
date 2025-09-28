# Simple LAMAP R Demo
# This script demonstrates the LAMAP algorithm with minimal example

# Load required library
library(raster)

# Source LAMAP functions
source("/home/azureuser/localfiles/otros/lamap/R/lamap.R")
source("/home/azureuser/localfiles/otros/lamap/R/utils.R")
source("/home/azureuser/localfiles/otros/lamap/R/jecdf.R")
source("/home/azureuser/localfiles/otros/lamap/R/weight.R")
source("/home/azureuser/localfiles/otros/lamap/R/unionIndependent.R")

cat("=== Simple LAMAP Demo ===\n")

# Create simple known site data
# Format: id, x, y, elevation, slope, dist_water
known_site_data <- data.frame(
  id = c(rep("Site1", 5), rep("Site2", 5), rep("Site3", 5)),
  x = c(100, 101, 102, 99, 103,   # Site 1 catchment
        150, 151, 149, 152, 148,  # Site 2 catchment  
        200, 201, 202, 199, 203), # Site 3 catchment
  y = c(100, 101, 99, 102, 98,    # Site 1 catchment
        150, 149, 151, 148, 152,  # Site 2 catchment
        200, 199, 201, 198, 202), # Site 3 catchment
  elevation = c(150, 155, 152, 148, 153,  # Site 1: moderate elevation
                180, 175, 185, 178, 182,  # Site 2: higher elevation
                120, 115, 125, 118, 122), # Site 3: lower elevation
  slope = c(0.5, 0.8, 0.6, 0.4, 0.7,     # Site 1: gentle slopes
            1.2, 1.5, 1.0, 1.3, 1.1,     # Site 2: steeper slopes
            0.2, 0.3, 0.1, 0.4, 0.2),    # Site 3: very gentle
  dist_water = c(200, 180, 220, 190, 210, # Site 1: near water
                 500, 480, 520, 490, 510, # Site 2: moderate distance
                 50, 40, 60, 45, 55)      # Site 3: very close to water
)

cat("Created 3 sites with 5 catchment samples each\n")
print(head(known_site_data, 9))

# Process for LAMAP
site_coords <- knownsiteCoords(known_site_data)
colnames(site_coords) <- c("id", "x", "y")
site_coords <- as.data.frame(site_coords)
site_coords$x <- as.numeric(site_coords$x)
site_coords$y <- as.numeric(site_coords$y)

cat("\nSite coordinates:\n")
print(site_coords)

# Create probability density functions
site_pcdfs <- knownsitePcdfs(known_site_data)

# Define integration steps
steps <- c(10, 0.5, 100)  # elevation, slope, dist_water

# Test 3 locations
test_locations <- data.frame(
  x = c(125, 175, 175),
  y = c(125, 175, 125),
  elevation = c(160, 190, 140),
  slope = c(0.6, 1.4, 0.8),
  dist_water = c(150, 400, 600)
)

cat("\n=== LAMAP Results ===\n")
for(i in 1:nrow(test_locations)) {
  test_loc <- test_locations[i, ]
  
  lamap_value <- lamap(
    observed = test_loc,
    knownsite_pcdfs = site_pcdfs,
    knownsite_coords = site_coords,
    steps = steps
  )
  
  cat(sprintf("Location %d: (%.0f,%.0f) LAMAP = %.4f\n", 
              i, test_loc$x, test_loc$y, lamap_value))
}

cat("\nDemo complete!\n")
