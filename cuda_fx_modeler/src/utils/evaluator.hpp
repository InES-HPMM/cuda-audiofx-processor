#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <path.hpp>
#include <rmsd.cuh>
#include <signal.hpp>
#include <sstream>
#include <vector>

#include "spdlog/spdlog.h"

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;

class StatisticValues {
   private:
    double getMean(const std::vector<double>& data_points) {
        return std::accumulate(data_points.begin(), data_points.end(), 0.0) / data_points.size();
    }
    double getStdev(const std::vector<double>& data_points, const double mean) {
        std::vector<double> diff(data_points.size());
        std::transform(data_points.begin(), data_points.end(), diff.begin(), [mean](double x) { return x - mean; });
        double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        return std::sqrt(sq_sum / data_points.size());
    }

    void getOutlierCounts(const std::vector<double>& data_points, const double critical_threshold, int& n_std1, int& n_std2, int& n_std3, int& n_critical) {
        auto std1_ = std1();
        auto std2_ = std2();
        auto std3_ = std3();
        auto is_gt_std1 = [std1_, std2_](auto const& val) { return std1_ <= val && val < std2_; };
        auto is_gt_std2 = [std2_, std3_](auto const& val) { return std2_ <= val && val < std3_; };
        auto is_gt_std3 = [std3_](auto const& val) { return std3_ <= val; };
        auto is_gt_critical = [critical_threshold](auto const& val) { return val >= critical_threshold; };
        auto lam = [is_gt_std1, is_gt_std2, is_gt_std3, is_gt_critical](auto a, auto const& val) {
            return std::array{
                a[0] += is_gt_std1(val),
                a[1] += is_gt_std2(val),
                a[2] += is_gt_std3(val),
                a[3] += is_gt_critical(val),
            };
        };
        auto res = std::accumulate(data_points.begin(), data_points.end(), std::array{0, 0, 0, 0}, lam);
        n_std1 = res[0];
        n_std2 = res[1];
        n_std3 = res[2];
        n_critical = critical_threshold > 0 ? res[3] : 0;
    }

   public:
    double max;
    double min;
    double mean;
    double mean_clipped;
    double stdev;
    double stdev_clipped;
    int n_data;
    int n_std1;
    int n_std2;
    int n_std3;
    int n_critical;

    StatisticValues() : max(0), min(0), mean(0), mean_clipped(0), stdev(0), stdev_clipped(0), n_data(0), n_std1(0), n_std2(0), n_std3(0), n_critical(0) {}
    StatisticValues(const std::vector<double>& execution_times, double critical_threshold, double clip_threshold = 0) {
        n_data = execution_times.size();
        max = *std::max_element(execution_times.begin(), execution_times.end());
        min = *std::min_element(execution_times.begin(), execution_times.end());
        mean = getMean(execution_times);
        stdev = getStdev(execution_times, mean);

        auto clip_threshold_ = clip_threshold == 0 ? std3() : clip_threshold;
        std::vector<double> clipped_execution_times;
        std::transform(execution_times.begin(), execution_times.end(), std::back_inserter(clipped_execution_times), [clip_threshold_](double val) { return val < clip_threshold_ ? val : clip_threshold_; });
        mean_clipped = getMean(clipped_execution_times);
        stdev_clipped = getStdev(clipped_execution_times, mean_clipped);

        getOutlierCounts(execution_times, critical_threshold, n_std1, n_std2, n_std3, n_critical);
    }
    double std1() { return mean + stdev; };
    double std2() { return mean + 2 * stdev; };
    double std3() { return mean + 3 * stdev; };
    double p_std1() { return 100.0 / n_data * (double)n_std1; }
    double p_std2() { return 100.0 / n_data * (double)n_std2; }
    double p_std3() { return 100.0 / n_data * (double)n_std3; }
    double p_critical() { return 100.0 / n_data * (double)n_critical; }

    void addValuesToColumnMajorMatrix(std::vector<std::vector<double>>& matrix, size_t row) {
        matrix[0][row] = max;
        matrix[1][row] = min;
        matrix[2][row] = mean;
        matrix[3][row] = mean_clipped;
        matrix[4][row] = stdev;
        matrix[5][row] = stdev_clipped;
        matrix[6][row] = std1();
        matrix[7][row] = std2();
        matrix[8][row] = std3();
        matrix[9][row] = n_std1;
        matrix[10][row] = n_std2;
        matrix[11][row] = n_std3;
        matrix[12][row] = n_critical;
        matrix[13][row] = p_std1();
        matrix[14][row] = p_std2();
        matrix[15][row] = p_std3();
        matrix[16][row] = p_critical();
    }
};

class PerformanceMeasurement {
   private:
    std::string _name;
    StatisticValues _stats;
    std::vector<double> _execution_times;
    double _buffer_duration;

   public:
    static void writeStatisticsToCsv(std::string path, std::vector<PerformanceMeasurement*> pms, bool append = false) {
        std::ofstream file;
        file.open(path, append ? std::ios::openmode::_S_app : std::ios::openmode::_S_trunc);
        if (!append || file.tellp() == 0) file << pms[0]->getCsvHeader() << std::endl;

        std::vector<std::vector<double>> matrix(17, std::vector<double>(pms.size()));
        for (size_t i = 0; i < pms.size(); i++) {
            file << pms[i]->getCsvLine() << std::endl;
            pms[i]->getStats().addValuesToColumnMajorMatrix(matrix, i);
        }
        if (!append) {
            file << ", ";
            for (size_t i = 0; i < matrix.size(); i++) {
                file << ",";
                auto min = min_element(matrix[i].begin(), matrix[i].end()) - matrix[i].begin();
                file << std::to_string(min);
            }
            file << std::endl;
        }
    }

    static void writeDataToCsv(std::string path, std::vector<PerformanceMeasurement*> pms, bool append = false) {
        std::ofstream file;
        file.open(path, append ? std::ios::openmode::_S_app : std::ios::openmode::_S_trunc);
        file << std::setprecision(20);
        for (size_t i = 0; i < pms.size(); i++) {
            file << pms[i]->getName();
            for (size_t j = 0; j < pms[i]->getData().size(); j++) {
                file << "," << pms[i]->getData()[j];
            }
            file << std::endl;
        }
    }

    PerformanceMeasurement(std::string name, const std::vector<double>& execution_times, double buffer_duration_us) {
        _name = name;
        _execution_times = execution_times;
        _buffer_duration = buffer_duration_us;
        _stats = StatisticValues(execution_times, buffer_duration_us);
    }

    StatisticValues getStats() { return _stats; }
    std::vector<double>& getData() { return _execution_times; }
    std::string getName() { return _name; }

    std::string getCsvHeader() {
        return "name,buffer_duration,max,min,mean,mean_c,stdev,stdev_c,std1,std2,std3,n_std1,n_std2,n_std3,n_critical,p_std1,p_std2,p_std3,p_critical";
    }
    std::string getCsvLine() {
        return _name + "," + std::to_string(_buffer_duration) + "," + std::to_string(_stats.max) + "," + std::to_string(_stats.min) + "," + std::to_string(_stats.mean) + "," +
               std::to_string(_stats.mean_clipped) + "," + std::to_string(_stats.stdev) + "," + std::to_string(_stats.stdev_clipped) + "," +
               std::to_string(_stats.std1()) + "," + std::to_string(_stats.std2()) + "," + std::to_string(_stats.std3()) + "," +
               std::to_string(_stats.n_std1) + "," + std::to_string(_stats.n_std2) + "," + std::to_string(_stats.n_std3) + "," + std::to_string(_stats.n_critical) + "," +
               std::to_string(_stats.p_std1()) + "," + std::to_string(_stats.p_std2()) + "," + std::to_string(_stats.p_std3()) + "," + std::to_string(_stats.p_critical());
    }

    void print() {
        spdlog::info("------------------------{}--------------------------------", _name);
        spdlog::info("Max:             {} us", _stats.max);
        spdlog::info("Min:             {} us", _stats.min);
        spdlog::info("Mean:            {} us", _stats.mean);
        spdlog::info("Mean Clipped:    {} us", _stats.mean_clipped);
        spdlog::info("Stdev:           {} us", _stats.stdev);
        spdlog::info("Stdev  Clipped:  {} us", _stats.stdev_clipped);
        spdlog::info("Buffer Duration: {} us", _buffer_duration);
        spdlog::info("STD1 ({}us) count: {}, percentage: {}", _stats.std1(), _stats.n_std1, _stats.p_std1());
        spdlog::info("STD2 ({}us) count: {}, percentage: {}", _stats.std2(), _stats.n_std2, _stats.p_std2());
        spdlog::info("STD3 ({}us) count: {}, percentage: {}", _stats.std3(), _stats.n_std3, _stats.p_std3());
        spdlog::info("Critical ({}us) count: {}, percentage: {}", _buffer_duration, _stats.n_critical, _stats.p_critical());
    }
};

class Evaluator {
   protected:
    size_t _n_warmup = 0;
    size_t _n_measure = 0;
    bool _simulate_buffer_intervals = false;
    size_t _n_proc_frames = 0;
    virtual void setup() = 0;
    virtual void process() = 0;
    virtual void postProcess() = 0;
    virtual void spin() { throw std::runtime_error("Not implemented"); };
    virtual void teardown() = 0;
    virtual IFPSignal* test(const IFPSignal* input) { throw std::runtime_error("Not implemented"); };

    virtual std::string getName() = 0;
    virtual std::string serialize_settings() {
        return getName() + "-nw" + std::to_string(_n_warmup) + "-nm" + std::to_string(_n_measure) + "-ns" + std::to_string(_n_proc_frames) + "-rt" + std::to_string(_simulate_buffer_intervals);
    }

    bool _testAccuracy(const IFPSignal* input, const IFPSignal* expected_output, const size_t n_proc_frames, const float max_rmsd, const int expected_min_rmsd_offset = 0, const bool write_output = false, const size_t max_rmsd_offset = 1000) {
        _n_proc_frames = n_proc_frames;

        setup();

        IFPSignal* actual_output = test(input);
        if (write_output) {
            actual_output->writeToWav(path::out(getName() + "_actual_output.wav"), BitDepth::BD_24);
        }

        std::vector<float> rmsds;
        std::vector<int> minimal_rmsd_offsets;
        size_t n_samples;
        switch (expected_output->getChannelOrder()) {
            case ChannelOrder::INTERLEAVED:
                minimal_rmsd_offsets.resize(1);
                n_samples = std::min(expected_output->getSampleCount(), actual_output->getSampleCount());
                rmsds.push_back(getMinimalRMSD((float*)expected_output->getDataPtrConst(), (float*)actual_output->getDataPtrMod(), n_samples, max_rmsd_offset, &minimal_rmsd_offsets[0]));
                break;
            case ChannelOrder::PLANAR:
                minimal_rmsd_offsets.resize(expected_output->getChannelCount());
                n_samples = std::min(expected_output->getFrameCount(), actual_output->getFrameCount());
                for (size_t c = 0; c < expected_output->getChannelCount(); c++) {
                    float* expected = ((float**)expected_output->getDataPtrConst())[c];
                    float* actual = ((float**)actual_output->getDataPtrConst())[c];
                    rmsds.push_back(getMinimalRMSD(expected, actual, n_samples, max_rmsd_offset, &minimal_rmsd_offsets[c]));
                }
                break;
            default:
                throw std::runtime_error("Unsupported channel order");
        }

        teardown();

        // Create a stringstream to hold the RMSD values
        std::ostringstream rmsd_stream;
        rmsd_stream << std::scientific << std::setprecision(20);
        for (size_t i = 0; i < rmsds.size(); ++i) {
            if (i != 0) {
                rmsd_stream << ", ";
            }
            rmsd_stream << rmsds[i];
        }
        std::string minimal_rmsd_offsets_str = std::accumulate(minimal_rmsd_offsets.begin(), minimal_rmsd_offsets.end(), std::string(""), [](std::string acc, int offset) { return acc + std::to_string(offset) + ", "; });

        if (std::any_of(minimal_rmsd_offsets.begin(), minimal_rmsd_offsets.end(), [expected_min_rmsd_offset](int offset) { return offset != expected_min_rmsd_offset; }))
            spdlog::warn("{}: minimal RMSD at offset {} not at {}", getName(), minimal_rmsd_offsets_str, expected_min_rmsd_offset);

        if (std::any_of(rmsds.begin(), rmsds.end(), [max_rmsd](float rmsd) { return std::isnan(rmsd) || rmsd > max_rmsd; })) {
            spdlog::error("{}: minimal RMSD at {} too high: {}", getName(), minimal_rmsd_offsets_str, rmsd_stream.str());
            return false;
        } else {
            spdlog::info("{}: minimal RMSD at {} within tolerance: {}", getName(), minimal_rmsd_offsets_str, rmsd_stream.str());
            return true;
        }
    }

    PerformanceMeasurement* _measurePerformance(size_t n_warmup, size_t n_measure, size_t n_proc_frames, bool simulate_buffer_intervals) {
        _n_warmup = n_warmup;
        _n_measure = n_measure;
        _simulate_buffer_intervals = simulate_buffer_intervals;
        _n_proc_frames = n_proc_frames;
        setup();
        for (size_t i = 0; i < _n_warmup; i++) {
            process();
        }

        std::vector<double> execution_times = std::vector<double>(_n_measure);
        for (size_t i = 0; i < _n_measure; i++) {
            auto start = high_resolution_clock::now();
            process();
            postProcess();
            auto end = high_resolution_clock::now();
            execution_times[i] = duration<double, std::micro>(end - start).count();
            if (simulate_buffer_intervals) {
                while (duration<double, std::micro>(high_resolution_clock::now() - start).count() < _n_proc_frames / 0.048) {
                    spin();
                }
            }
        }
        teardown();
        PerformanceMeasurement* pm = new PerformanceMeasurement(serialize_settings(), execution_times, _n_proc_frames / 0.048);
        // pm->print();
        return pm;
    }

   public:
    virtual ~Evaluator() {}

    // void gridSearchPerformance(size_t n_warmup, size_t n_measure, std::vector<size_t> n_proc_frames, bool simulate_buffer_intervals) {
    //     std::vector<PerformanceMeasurement*> pms;
    //     for (auto n : n_proc_frames) {
    //         pms.push_back(measurePerformance(n_warmup, n_measure, n, simulate_buffer_intervals));
    //     }
    // }
};
