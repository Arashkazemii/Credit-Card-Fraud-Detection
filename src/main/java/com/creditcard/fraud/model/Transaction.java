package com.creditcard.fraud.model;

import java.time.Instant;

public class Transaction {
    private Instant time;
    private double[] features; // V1-V28
    private double amount;
    private int isFraud; // 0: legitimate, 1: fraudulent

    public Transaction(Instant time, double[] features, double amount, int isFraud) {
        this.time = time;
        this.features = features;
        this.amount = amount;
        this.isFraud = isFraud;
    }

    // Getters and setters
    public Instant getTime() {
        return time;
    }

    public void setTime(Instant time) {
        this.time = time;
    }

    public double[] getFeatures() {
        return features;
    }

    public void setFeatures(double[] features) {
        this.features = features;
    }

    public double getAmount() {
        return amount;
    }

    public void setAmount(double amount) {
        this.amount = amount;
    }

    public int getIsFraud() {
        return isFraud;
    }

    public void setIsFraud(int isFraud) {
        this.isFraud = isFraud;
    }
} 