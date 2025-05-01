package com.creditcard.fraud.util;

import com.creditcard.fraud.model.Transaction;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;

import java.util.List;

public class StatisticsAnalyzer {
    public static void analyzeTransactions(List<Transaction> transactions) {
        // Amount statistics
        DescriptiveStatistics amountStats = new DescriptiveStatistics();
        DescriptiveStatistics fraudAmountStats = new DescriptiveStatistics();
        DescriptiveStatistics legitAmountStats = new DescriptiveStatistics();

        // Feature correlations
        double[][] featureMatrix = new double[transactions.size()][28];
        double[] amounts = new double[transactions.size()];

        for (int i = 0; i < transactions.size(); i++) {
            Transaction t = transactions.get(i);
            double amount = t.getAmount();
            amountStats.addValue(amount);

            if (t.getIsFraud() == 1) {
                fraudAmountStats.addValue(amount);
            } else {
                legitAmountStats.addValue(amount);
            }

            // Prepare data for correlation analysis
            System.arraycopy(t.getFeatures(), 0, featureMatrix[i], 0, 28);
            amounts[i] = amount;
        }

        // Print statistics
        System.out.println("\nTransaction Amount Statistics:");
        System.out.printf("Mean amount: $%.2f%n", amountStats.getMean());
        System.out.printf("Median amount: $%.2f%n", amountStats.getPercentile(50));
        System.out.printf("Max amount: $%.2f%n", amountStats.getMax());
        System.out.printf("Min amount: $%.2f%n", amountStats.getMin());

        System.out.println("\nFraudulent Transaction Amount Statistics:");
        System.out.printf("Mean amount: $%.2f%n", fraudAmountStats.getMean());
        System.out.printf("Median amount: $%.2f%n", fraudAmountStats.getPercentile(50));

        System.out.println("\nLegitimate Transaction Amount Statistics:");
        System.out.printf("Mean amount: $%.2f%n", legitAmountStats.getMean());
        System.out.printf("Median amount: $%.2f%n", legitAmountStats.getPercentile(50));

        // Calculate correlations between amount and features
        PearsonsCorrelation correlation = new PearsonsCorrelation();
        System.out.println("\nTop 5 Features Correlated with Amount:");
        double[] correlations = new double[28];
        for (int i = 0; i < 28; i++) {
            double[] feature = new double[transactions.size()];
            for (int j = 0; j < transactions.size(); j++) {
                feature[j] = featureMatrix[j][i];
            }
            correlations[i] = correlation.correlation(feature, amounts);
        }

        // Print top 5 correlations
        for (int i = 0; i < 5; i++) {
            int maxIndex = 0;
            double maxCorr = 0;
            for (int j = 0; j < 28; j++) {
                if (Math.abs(correlations[j]) > Math.abs(maxCorr)) {
                    maxCorr = correlations[j];
                    maxIndex = j;
                }
            }
            System.out.printf("V%d: %.4f%n", maxIndex + 1, maxCorr);
            correlations[maxIndex] = 0; // Mark as processed
        }
    }
}