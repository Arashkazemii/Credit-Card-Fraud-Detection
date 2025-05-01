package com.creditcard.fraud;

import com.creditcard.fraud.model.Transaction;
import com.creditcard.fraud.util.StatisticsAnalyzer;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class BatchProcessor {
    private static final Logger logger = LoggerFactory.getLogger(BatchProcessor.class);
    private static final int NUM_FEATURES = 28; // V1-V28

    public static void main(String[] args) {
        if (args.length != 1) {
            logger.error("Usage: java -jar batch-processing.jar <input_csv_file>");
            System.exit(1);
        }

        String inputFile = args[0];
        try {
            processBatch(inputFile);
        } catch (IOException e) {
            logger.error("Error processing batch: {}", e.getMessage());
            System.exit(1);
        }
    }

    public static void processBatch(String inputFile) throws IOException {
        logger.info("Starting batch processing of file: {}", inputFile);

        List<Transaction> transactions = new ArrayList<>();
        AtomicInteger fraudCount = new AtomicInteger(0);
        AtomicInteger totalCount = new AtomicInteger(0);

        try (CSVParser parser = new CSVParser(new FileReader(inputFile),
                CSVFormat.DEFAULT.withFirstRecordAsHeader())) {

            for (CSVRecord record : parser) {
                totalCount.incrementAndGet();

                // Parse time
                long timeSeconds = Long.parseLong(record.get("Time"));
                Instant time = Instant.ofEpochSecond(timeSeconds);

                // Parse features V1-V28
                double[] features = new double[NUM_FEATURES];
                for (int i = 0; i < NUM_FEATURES; i++) {
                    features[i] = Double.parseDouble(record.get("V" + (i + 1)));
                }

                // Parse amount and class
                double amount = Double.parseDouble(record.get("Amount"));
                int isFraud = Integer.parseInt(record.get("Class"));

                if (isFraud == 1) {
                    fraudCount.incrementAndGet();
                }

                transactions.add(new Transaction(time, features, amount, isFraud));
            }
        }

        // Calculate statistics
        double fraudPercentage = (double) fraudCount.get() / totalCount.get() * 100;

        logger.info("Batch processing completed");
        logger.info("Total transactions processed: {}", totalCount.get());
        logger.info("Fraudulent transactions: {} ({:.2f}%)",
                fraudCount.get(), fraudPercentage);

        // Perform detailed statistical analysis
        StatisticsAnalyzer.analyzeTransactions(transactions);
    }
}