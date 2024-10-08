

#pragma once

float getRMSD(const float* expected, const float* actual, const size_t n_samples, const size_t n_channels);
float getMinimalRMSD(const float* expected, const float* actual, const size_t size, const size_t max_offset, int* minimal_rmsd_offset = nullptr);
bool testMinimalRMSDIntegrity();
