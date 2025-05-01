# Java Batch Processing Component

This component provides batch processing capabilities for the Credit Card Fraud Detection system. It's designed to handle large-scale historical data analysis efficiently.

## Features

- High-performance CSV processing
- Statistical analysis of transaction data
- Correlation analysis between features
- Detailed transaction amount statistics
- Separate analysis for fraudulent and legitimate transactions

## Building

1. Make sure you have Java 11 and Maven installed
2. Navigate to the project root directory
3. Run:
```bash
mvn clean package
```

This will create a fat JAR file in the `target` directory.

## Usage

Run the batch processor with:
```bash
java -jar target/batch-processing-1.0-SNAPSHOT-jar-with-dependencies.jar <input_csv_file>
```

The processor will:
1. Read and parse the CSV file
2. Calculate basic statistics (total transactions, fraud percentage)
3. Perform detailed statistical analysis including:
   - Transaction amount statistics
   - Separate analysis for fraudulent and legitimate transactions
   - Feature correlations with transaction amounts

## Output

The processor outputs:
- Total number of transactions processed
- Number and percentage of fraudulent transactions
- Mean, median, max, and min transaction amounts
- Separate statistics for fraudulent and legitimate transactions
- Top 5 features most correlated with transaction amounts

## Integration

This component can be integrated with the main Python application by:
1. Running the batch processor on historical data
2. Exporting the results to a format readable by the Python application
3. Using the insights to improve the machine learning models 