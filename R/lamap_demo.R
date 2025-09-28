# R LAMAP Demo - Same Data as Python Demo
# This creates identical synthetic data to the Python notebook for direct comparison

library(raster)

# Source LAMAP functions
source("./R/lamap.R")
source("./R/utils.R")
source("./R/jecdf.R")
source("./R/weight.R")
source("./R/unionIndependent.R")

cat("=== R LAMAP Demo - Same Data as Python ===\n")

# Create synthetic terrain (same as Python: 80x100 grid)
H <- 80
W <- 100

# Create coordinate grids (same as Python)
ygrid <- outer(rep(1, W), 1:H - 0.5)
xgrid <- outer(1:W - 0.5, rep(1, H))

# Generate elevation (same formula as Python)
elev <- 120 + 0.35 * xgrid + 0.15 * ygrid + 8 * sin(xgrid/15.0) + 5 * cos(ygrid/21.0)

# Calculate slope (simplified gradient)
slope <- matrix(0, nrow = H, ncol = W)
for(i in 2:(H-1)) {
  for(j in 2:(W-1)) {
    dx <- (elev[i, j+1] - elev[i, j-1]) / 2
    dy <- (elev[i+1, j] - elev[i-1, j]) / 2
    slope[i, j] <- sqrt(dx^2 + dy^2)
  }
}

# Create synthetic river and distance to water
river_y <- H/2 + 10 * sin(seq(0, 3*pi, length.out = W))
d2w <- matrix(0, nrow = H, ncol = W)
for(i in 1:H) {
  for(j in 1:W) {
    d2w[i, j] <- abs(i - river_y[j])  # Distance to river
  }
}

cat("Created synthetic terrain: 80x100 grid\n")
cat(sprintf("Elevation range: %.1f - %.1f\n", min(elev), max(elev)))
cat(sprintf("Slope range: %.2f - %.2f\n", min(slope), max(slope)))
cat(sprintf("Distance to water range: %.1f - %.1f\n", min(d2w), max(d2w)))

# Create the same 5 sites as Python (A, B, C, D, E)
sites <- data.frame(
  id = c('A', 'B', 'C', 'D', 'E'),
  x = c(20.0, 70.0, 50.0, 30.0, 85.0),
  y = c(25.0, 35.0, 60.0, 10.0, 65.0)
)

# Generate catchment data around each site (radius = 15)
catchment_radius <- 15
known_site_data <- data.frame()

for(site_idx in 1:nrow(sites)) {
  site_x <- sites$x[site_idx]
  site_y <- sites$y[site_idx]
  site_id <- sites$id[site_idx]
  
  # Sample within catchment radius
  for(dx in -catchment_radius:catchment_radius) {
    for(dy in -catchment_radius:catchment_radius) {
      if(sqrt(dx^2 + dy^2) <= catchment_radius) {
        sample_x <- round(site_x) + dx
        sample_y <- round(site_y) + dy
        
        # Check bounds
        if(sample_x >= 1 && sample_x <= W && sample_y >= 1 && sample_y <= H) {
          site_sample <- data.frame(
            id = site_id,
            x = sample_x,
            y = sample_y,
            elevation = round(elev[sample_y, sample_x], 1),
            slope = round(slope[sample_y, sample_x], 2),
            dist_water = round(d2w[sample_y, sample_x], 1)
          )
          known_site_data <- rbind(known_site_data, site_sample)
        }
      }
    }
  }
}

cat(sprintf("Created %d sites with %d catchment samples total\n", 
            nrow(sites), nrow(known_site_data)))

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

# Define integration steps (same as Python: eps values)
steps <- c(10.0, 0.1, 10.0)  # elevation, slope, dist_water
cat("Integration steps:", steps, "\n")

# Test a few specific locations
test_locations <- data.frame(
  x = c(25, 50, 75),
  y = c(30, 45, 60),
  elevation = c(elev[30, 25], elev[45, 50], elev[60, 75]),
  slope = c(slope[30, 25], slope[45, 50], slope[60, 75]),
  dist_water = c(d2w[30, 25], d2w[45, 50], d2w[60, 75])
)

cat("\n=== LAMAP Results (R Implementation) ===\n")
for(i in 1:nrow(test_locations)) {
  test_loc <- test_locations[i, ]
  
  lamap_value <- lamap(
    observed = test_loc,
    knownsite_pcdfs = site_pcdfs,
    knownsite_coords = site_coords,
    steps = steps,
    maxsites = 5,
    weightfun = "exponential",
    weightparams = c(0.25, 100)
  )
  
  cat(sprintf("Location %d: (%.0f,%.0f) LAMAP = %.4f\n", 
              i, test_loc$x, test_loc$y, lamap_value))
}

cat("\nDemo complete!\n")
