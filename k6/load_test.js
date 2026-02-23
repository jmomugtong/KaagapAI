/**
 * MedQuery Load Test
 * 
 * k6 load testing script for the MedQuery API.
 * Tests the query endpoint under various load conditions.
 * 
 * Run with: k6 run k6/load_test.js
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// ============================================
// Custom Metrics
// ============================================

const errorRate = new Rate('errors');
const queryLatency = new Trend('query_latency');
const cacheHitRate = new Rate('cache_hits');

// ============================================
// Test Configuration
// ============================================

export const options = {
    // Staged load test
    stages: [
        { duration: '30s', target: 2 },   // Warm up: ramp to 2 users
        { duration: '1m', target: 5 },    // Ramp up to 5 users
        { duration: '2m', target: 10 },   // Stay at 10 users (target from PRD)
        { duration: '1m', target: 10 },   // Stay at peak
        { duration: '30s', target: 0 },   // Ramp down
    ],

    // Performance thresholds
    thresholds: {
        // 95% of requests must complete under 2s (PRD requirement)
        'http_req_duration{expected_response:true}': ['p(95)<2000'],

        // Error rate must be less than 1%
        'http_req_failed': ['rate<0.01'],

        // Custom query latency threshold
        'query_latency': ['p(95)<2000', 'avg<1000'],

        // Error rate threshold
        'errors': ['rate<0.05'],
    },
};

// ============================================
// Test Data
// ============================================

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

const SAMPLE_QUERIES = [
    "What is the post-operative pain protocol for knee replacement?",
    "What painkillers are safe for cardiac patients on aspirin?",
    "What are the signs of post-surgical infection?",
    "What is the recommended dosage of acetaminophen for adults?",
    "When should a patient be discharged after knee surgery?",
    "What are the contraindications for ibuprofen use?",
    "How often should vital signs be monitored post-surgery?",
    "What is the protocol for managing post-operative nausea?",
    "What are the warning signs of deep vein thrombosis?",
    "How should wound care be performed after surgery?",
];

// ============================================
// Helper Functions
// ============================================

function getRandomQuery() {
    return SAMPLE_QUERIES[Math.floor(Math.random() * SAMPLE_QUERIES.length)];
}

// ============================================
// Test Scenarios
// ============================================

export default function (data) {
    const authHeader = data.token ? { 'Authorization': `Bearer ${data.token}` } : {};

    group('Query Endpoint Load Test', function () {
        const query = {
            question: getRandomQuery(),
            max_results: 5,
            confidence_threshold: 0.70,
        };

        const params = {
            headers: Object.assign({
                'Content-Type': 'application/json',
            }, authHeader),
            tags: { name: 'query' },
        };

        const startTime = Date.now();

        const response = http.post(
            `${BASE_URL}/api/v1/query`,
            JSON.stringify(query),
            params
        );

        const endTime = Date.now();
        const latency = endTime - startTime;

        // Record custom metrics
        queryLatency.add(latency);

        // Check response
        const success = check(response, {
            'status is 200': (r) => r.status === 200,
            'response has answer': (r) => {
                const body = r.json();
                return body && body.answer !== undefined;
            },
            'response has citations': (r) => {
                const body = r.json();
                return body && Array.isArray(body.citations);
            },
            'response has confidence': (r) => {
                const body = r.json();
                return body && typeof body.confidence === 'number';
            },
            'latency under 2s': (r) => r.timings.duration < 2000,
        });

        // Track errors
        errorRate.add(!success);

        // Check for cache hit (when implemented)
        if (response.status === 200) {
            const body = response.json();
            if (body && body.cache_hit !== undefined) {
                cacheHitRate.add(body.cache_hit);
            }
        }

        // Simulate user think time
        sleep(Math.random() * 2 + 1);  // 1-3 seconds
    });
}

// ============================================
// Health Check Scenario
// ============================================

export function healthCheck() {
    group('Health Check', function () {
        const response = http.get(`${BASE_URL}/health`);

        check(response, {
            'health check returns 200': (r) => r.status === 200,
            'health status is healthy': (r) => r.json().status === 'healthy',
        });
    });
}

// ============================================
// Auth Endpoint Scenario
// ============================================

export function authTest() {
    group('Auth Endpoints', function () {
        // Register a unique user
        const email = `k6_auth_${__VU}_${__ITER}_${Date.now()}@test.com`;
        const regPayload = JSON.stringify({
            email: email,
            password: 'TestPassword123!',
            full_name: 'Auth Test User',
        });

        const regRes = http.post(`${BASE_URL}/api/v1/auth/register`, regPayload, {
            headers: { 'Content-Type': 'application/json' },
        });

        check(regRes, {
            'register returns 200': (r) => r.status === 200,
            'register returns token': (r) => r.json().access_token !== undefined,
        });

        // Login with the same credentials
        const loginPayload = JSON.stringify({
            email: email,
            password: 'TestPassword123!',
        });

        const loginRes = http.post(`${BASE_URL}/api/v1/auth/login`, loginPayload, {
            headers: { 'Content-Type': 'application/json' },
        });

        check(loginRes, {
            'login returns 200': (r) => r.status === 200,
            'login returns token': (r) => r.json().access_token !== undefined,
        });

        // Use the token to make an authenticated query
        if (loginRes.status === 200) {
            const token = loginRes.json().access_token;
            const queryRes = http.post(
                `${BASE_URL}/api/v1/query`,
                JSON.stringify({ question: 'What is the pain management protocol?' }),
                { headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` } }
            );

            check(queryRes, {
                'authenticated query returns 200': (r) => r.status === 200,
            });
        }
    });
}

// ============================================
// Setup and Teardown
// ============================================

export function setup() {
    // Verify API is available before starting test
    const healthRes = http.get(`${BASE_URL}/health`);

    if (healthRes.status !== 200) {
        throw new Error(`API health check failed: ${healthRes.status}`);
    }

    console.log('API is healthy, registering test user...');

    // Register a test user and obtain a JWT token
    const uniqueEmail = `k6_loadtest_${Date.now()}@test.com`;
    const regPayload = JSON.stringify({
        email: uniqueEmail,
        password: 'K6LoadTest!2024',
        full_name: 'k6 Load Tester',
    });

    const regRes = http.post(`${BASE_URL}/api/v1/auth/register`, regPayload, {
        headers: { 'Content-Type': 'application/json' },
    });

    let token = null;
    if (regRes.status === 200 || regRes.status === 201) {
        token = regRes.json().access_token;
        console.log('Test user registered, token obtained.');
    } else {
        console.warn(`Registration returned ${regRes.status} — running without auth.`);
    }

    return {
        startTime: new Date().toISOString(),
        token: token,
    };
}

export function teardown(data) {
    console.log(`Load test completed. Started at: ${data.startTime}`);
}

// ============================================
// Custom Summary
// ============================================

export function handleSummary(data) {
    // Generate a custom summary report
    const summary = {
        timestamp: new Date().toISOString(),
        metrics: {
            requests: data.metrics.http_reqs.values.count,
            errors: data.metrics.http_req_failed.values.rate,
            latency: {
                avg: data.metrics.http_req_duration.values.avg,
                p50: data.metrics.http_req_duration.values['p(50)'],
                p95: data.metrics.http_req_duration.values['p(95)'],
                p99: data.metrics.http_req_duration.values['p(99)'],
            },
        },
        passed: Object.values(data.root_group.checks).every(c => c.passes === c.fails + c.passes),
    };

    return {
        'stdout': textSummary(data, { indent: ' ', enableColors: true }),
        'k6/results/summary.json': JSON.stringify(summary, null, 2),
    };
}

// Import text summary helper
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.1/index.js';
