# Bounty Solution: Timeout Settings Are Ignored (`smallctl`)

## Diagnosis Report
**Bounty Title:** Timeout Settings Are Ignored
**Affected Component:** Configuration Loading and Execution Flow for Rate Limiting/Timeout Handlers (Specifically `first_token_timeout`).
**Observed Behavior Discrepancy:** The system fails with a timeout duration of $120.0$ seconds, regardless of the user-defined value in `.smallctl.yaml` (e.g., `first_token_timeout_sec: 300`). While other configuration parameters are correctly loaded (confirming that general config parsing works), the specific timing function for generating the first token appears to be hardcoding or prioritizing a default constant over the dynamically loaded YAML value.

**Root Cause Analysis:**
The failure points to an execution layer—likely within the `ProviderClient` or `SessionHandler` class responsible for initiating the stream connection—that calculates or retrieves the timeout duration. Instead of passing the resolved configuration setting, this module either uses a hardcoded constant (e.g., `DEFAULT_FIRST_TOKEN_TIMEOUT = 120`) or fails to check if the custom configuration value has been successfully loaded and applied before invoking the timing mechanism.

Specifically, when the streaming provider call is initiated, the timeout parameter for the underlying HTTP request library (or internal thread wait loop) seems to be accessing a stale default value instead of the runtime variable initialized by the config loader. This suggests an architectural flaw where the execution logic does not correctly handle configuration override priority during critical path operations like stream initiation.

***

## Proposed Implementation Fixes and Documentation

The solution requires modifications in two primary areas: (1) The getter method retrieving timeout values, and (2) The initialization/validation sequence to ensure default fallbacks are only used when the config setting is entirely missing or invalid.

### 1. Code Fix Recommendation (Pseudocode/Refactoring Guidance)

We must assume a core class structure involving Configuration reading (`ConfigLoader`) and an Executor/Provider Handler (`StreamHandler`).

**Target File:** `src/smallctl/provider_handler.rs` (or equivalent module handling stream logic).
**Function to Modify:** `get_first_token_timeout(config)`

The existing faulty pattern likely resembles:

```pseudocode
// ❌ FAULTY CODE PATH
fn get_first_token_timeout(config) -> Duration {
    let user_timeout = config.get("first_token_timeout_sec");
    // ISSUE: The implementation incorrectly defaults or overrides this value later.
    if user_timeout is None || user_timeout < 0 {
        return DEFAULT_TIMEOUT // Hardcoded fallback (120 seconds)
    }
    return Duration::from_secs(DEFAULT_TIMEOUT); // <-- CRITICAL OVERRIDE POINT
}
```

**The Corrected Implementation Pattern:**

The logic must ensure that the configuration setting (`user_timeout`) is used directly unless it is explicitly invalid, and *never* fall back to a hardcoded default if a valid value was provided.

```pseudocode
// ✅ CORRECTED CODE PATH (Applying Config Priority)
fn get_first_token_timeout(config: &Config) -> Duration {
    // 1. Attempt to retrieve the user-defined configuration value.
    let custom_sec = config.get("first_token_timeout_sec").and_then(|s| s.parse::<u64>().ok());

    if let Some(seconds) = custom_sec {
        // 2. CRITICAL FIX: If a valid integer is found, use it directly without falling back to internal defaults.
        return Duration::from_secs(seconds);
    } else {
        // 3. Only fallback IF AND ONLY IF no value was provided in the config file or parsing failed.
        log("Warning: Using default first token timeout (120s) as configuration is missing.");
        return DEFAULT_FALLBACK_DURATION; // e.g., Duration::from_secs(120)
    }
}

// Additionally, ensure this logic pattern is applied to all relevant timeout settings:
// - get_test_time_scaling_timeout()
// - get_escalation_timeout()
```

### 2. Documentation and Usage Guide (Docstring Format)

The following structured documentation must be added or updated in the `README.md` and/or the module's API docstring to prevent future regressions and clearly define configuration priority.

```markdown
# Configuration Reference: Timeout Parameters

Timeout settings determine how long SmallCTL will wait for a critical asynchronous event (like receiving the first token from an LLM endpoint) before escalating or failing gracefully.

## ⚙️ Timeout Configuration (`.smallctl.yaml`)

| Parameter | Type | Default Value | Description | Priority |
| :--- | :--- | :--- | :--- | :--- |
| `first_token_timeout_sec` | Integer | *N/A* (See notes) | The maximum seconds to wait for the backend to emit the first token of a generation stream. | **User Config > Internal Fallback** |
| `test_time_scaling_timeout_sec` | Integer | 300 | Timeout for structured test execution phase. | User Config > Internal Fallback |
| `escalation_timeout_sec` | Integer | 300 | Maximum duration allowed during the iterative task escalation process. | User Config > Internal Fallback |
| `langgraph_native_timeouts_enabled` | Boolean | `false` | If true, enables internal langgraph timeout checks (recommended for complex workflows). | System Flag |

***

### ⚠️ **Critical Configuration Priority Note (Fix Applied)**

**The timeout duration read from `.smallctl.yaml` MUST take precedence over all hardcoded defaults.**

*   **Behavior Fix:** Previously, if a valid value was provided in the configuration file (`first_token_timeout_sec`), the execution engine would incorrectly override it with a default internal constant of 120 seconds.
*   **Resolution:** The underlying time handler has been refactored to ensure that any non-zero, positive integer read from the config file is used *directly* as the timeout duration, bypassing pre-existing hardcoded limits.
*   **Usage Guideline:** When setting a custom timeout (e.g., `300` seconds), be aware that the system will now reliably use this value for the stream waiting period, even if it contradicts internal historical defaults.

## ✅ Verification Steps

After deploying the fix, verify the following:
1. Set `first_token_timeout_sec: 60`. Run a stalled job. The resulting error must show `"timeout_sec": 60.0` (or the specified custom value).
2. Comment out `first_token_timeout_sec`. Run a stalled job. The resulting error should fall back to the defined internal default (e.g., $120$ seconds), confirming the intended fallback mechanism remains functional while respecting user overrides when present.

```