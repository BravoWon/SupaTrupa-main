# Troubleshooting & FAQ

**Solutions to Common Problems and Questions**

---

## Quick Troubleshooting

### System Won't Start

| Symptom | Check | Fix |
|---------|-------|-----|
| Black screen | Power | Verify power cable, try different outlet |
| Logo stuck | Boot process | Wait 5 minutes, then hard restart |
| Error on startup | Log files | Note error code, contact tech support |
| Slow to load | Network | Check network cable connection |

### No Data Displaying

| Symptom | Check | Fix |
|---------|-------|-----|
| All parameters show "--" | Data feed | Check WITS/WITSML connection |
| Some parameters blank | Specific sensors | Check individual sensor feeds |
| Data frozen | Update time | Look at timestamp - if old, restart |
| Wrong values | Source configuration | Verify correct well/rig selected |

### Display Issues

| Symptom | Check | Fix |
|---------|-------|-----|
| 3D view frozen | Graphics | Refresh page (F5) |
| Text too small | Zoom level | Ctrl/Cmd + to increase |
| Colors look wrong | Monitor | Check monitor color settings |
| Screen flickering | Refresh rate | Try different browser/restart |

---

## Regime Detection Issues

### Confidence Stays Low

**Why this happens**:
- System hasn't seen enough data yet
- Formation transition in progress
- Unusual drilling parameters
- Data quality issues

**What to do**:
1. Wait 2-3 stands for calibration
2. Check if all data inputs are present
3. Enter formation tops if missing
4. If persists, note it for tech support review

### Regime Changes Too Frequently

**Why this happens**:
- Actual instability in drilling
- Sensor noise
- Window size may need adjustment

**What to do**:
1. Check if drilling is actually unstable (ask driller)
2. Verify sensor readings are smooth
3. Note frequency - if every few seconds, may be sensor issue
4. Contact tech support if problematic

### Regime Never Changes (Stuck on Normal)

**Why this happens**:
- Actually drilling normally (good!)
- Data not updating
- Algorithm parameters need adjustment

**What to do**:
1. Verify data is updating (timestamps changing)
2. Intentionally induce minor stick-slip to test (with supervisor OK)
3. Check Betti numbers are responding to parameter changes

### Wrong Regime Detected

**Why this happens**:
- Unusual parameter combinations
- Algorithm hasn't seen this pattern before
- Data quality issue

**What to do**:
1. Log the event with what you think it actually is
2. Continue with your experienced judgment
3. Report to tech support for algorithm improvement

---

## Recommendation Issues

### Recommendation Doesn't Work

**Why this happens**:
- Multiple issues present
- Recommendation not applied long enough
- Different root cause than detected

**What to do**:
1. Ensure you applied the full recommendation (full WOB reduction, etc.)
2. Wait at least 60 seconds for effect
3. Try secondary recommendation
4. If nothing works, investigate other causes

### Recommendation Seems Wrong

**Why this happens**:
- Context system doesn't have (planned action, known issue, etc.)
- Unusual situation
- Possible algorithm limitation

**What to do**:
1. Trust your experience if low confidence
2. Log the event with your reasoning
3. Report for algorithm review if pattern repeats

### No Recommendation Given

**Why this happens**:
- Confidence too low to recommend
- System in learning mode
- Unknown pattern

**What to do**:
1. Check confidence level - if <70%, system isn't confident enough
2. Use your judgment
3. Note what you did and the outcome

---

## Data Entry Issues

### Can't Add Formation Top

**Symptom**: Formation button doesn't respond or errors

**Fix**:
1. Refresh the page
2. Try again
3. Check you're entering valid depth (within current well range)
4. Contact tech support if persists

### BHA Builder Not Saving

**Symptom**: BHA changes don't persist

**Fix**:
1. Click "Set Active BHA" after making changes
2. Verify save confirmation appears
3. If no confirmation, refresh and try again
4. Check you have write permissions

### Mission File Won't Load

**Symptom**: Error when opening .mission file

**Fix**:
1. Verify file isn't corrupted (can you see file size?)
2. Try opening on different computer
3. Contact tech support with file for recovery attempt

---

## Performance Issues

### System Running Slowly

**Check**:
1. How much data loaded? (Large mission files = slower)
2. Other programs running?
3. Network connection speed?

**Fix**:
1. Close unnecessary browser tabs
2. Close other applications
3. Try restarting browser
4. If persists, restart computer

### 3D View Laggy

**Check**:
1. Graphics card capabilities
2. Number of data points displayed

**Fix**:
1. Reduce point display density in settings
2. Disable auto-rotate
3. Try different browser (Chrome often fastest)

---

## Frequently Asked Questions

### General

**Q: How often does the system update?**
A: Every 1 second when connected to live data feed.

**Q: How far back does it look?**
A: Default is 30 seconds (30 samples at 1 Hz). This can be configured.

**Q: Does it work offline?**
A: It can replay saved mission files offline, but needs live data for real-time monitoring.

**Q: Is my data stored somewhere?**
A: Data is stored locally in mission files. No data leaves your computer unless you explicitly export/share.

### Technical

**Q: What's the minimum data it needs?**
A: WOB, RPM, and Torque minimum. More parameters = better detection.

**Q: Can it use MWD vibration data?**
A: Yes, if available via WITSML. Improves detection accuracy.

**Q: How does it handle data dropouts?**
A: Short dropouts (<5 seconds) are interpolated. Longer dropouts lower confidence.

**Q: Can I adjust the sensitivity?**
A: Yes, via Settings. But recommend leaving defaults unless you understand implications.

### Operational

**Q: Should I always follow the recommendation?**
A: Use it as input. High confidence (>85%) recommendations have high success rate. Lower confidence = use judgment.

**Q: What if I disagree with the system?**
A: Trust your experience, especially with low confidence recommendations. Log the event for future improvement.

**Q: How long does it take to learn a new well?**
A: Typically 2-3 stands for basic calibration, 1-2 days for good adaptation.

**Q: Does it work in all formations?**
A: Works in any formation. Some formations have more distinct signatures than others.

**Q: Can it predict problems before they start?**
A: It detects problems in early stages, often before noticeable symptoms. Not true prediction, but very early detection.

### Practical

**Q: What do I tell the relief driller at shift change?**
A: Current regime, confidence, any active recommendations, any events during your shift.

**Q: How do I know if it's working?**
A: Regime should respond to drilling changes. If you induce slight stick-slip, β₁ should increase.

**Q: What if power goes out?**
A: System has auto-save. On restart, it recovers last saved state (within 5 minutes of outage).

**Q: Who do I call if something breaks?**
A: Tech support number should be posted at the station. If not: _______________

---

## Error Messages

### Common Error Codes

| Error | Meaning | Fix |
|-------|---------|-----|
| `E001` | No data connection | Check network cable |
| `E002` | Data format error | Verify WITSML configuration |
| `E003` | Mission file corrupt | Try backup file |
| `E004` | Memory limit reached | Close and restart |
| `E005` | Algorithm timeout | Report to tech support |
| `E010` | License issue | Contact administrator |

### "Connection Lost" Message

1. Check physical network cable
2. Verify data source is running
3. Check firewall settings
4. Try restarting the application

### "Invalid Data" Warning

1. Check which parameter is flagged
2. Verify sensor is working
3. System will exclude invalid data from analysis
4. If critical parameter, fix sensor issue

### "Calibration Required" Notice

1. Normal for new well start
2. Enter formation tops
3. Verify BHA is set correctly
4. Allow 2-3 stands for calibration

---

## Reset Procedures

### Soft Reset (Try First)

1. Press F5 to refresh the browser
2. Wait for reload
3. Data should reconnect automatically

### Application Restart

1. Close browser completely
2. Wait 10 seconds
3. Reopen browser
4. Navigate to system URL

### Full System Restart

1. Save any unsaved work
2. Close all applications
3. Restart computer
4. Reopen system after boot

### Factory Reset (Last Resort)

**WARNING**: This clears all settings and mission files.

1. Contact tech support first
2. Back up mission files if possible
3. Follow tech support instructions

---

## Maintenance

### Daily

- Verify data is updating
- Check regime display is responding
- Review any overnight alerts

### Weekly

- Export event logs
- Verify disk space adequate
- Check for software updates

### Per Well

- Save and archive mission file
- Export summary report
- Note any algorithm feedback

---

## Contact Information

### Tech Support

- Phone: _______________
- Email: _______________
- Hours: _______________
- Emergency (24/7): _______________

### On-Site Support

- Contact: _______________
- Location: _______________

### Training

- Request form: _______________
- Contact: _______________

---

## Quick Reference: When To Call Support

**Call Immediately**:
- System crashes repeatedly
- Data not updating for >5 minutes
- Error codes E005 or E010

**Call During Business Hours**:
- Recommendations consistently wrong
- Confidence stays low after calibration
- Feature requests

**Don't Need To Call (Handle Locally)**:
- Brief data dropouts (<1 minute)
- Slow performance (try restart)
- Formation entry questions (see this guide)
