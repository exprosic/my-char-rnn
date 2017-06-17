/**
 * Created by exprosic on 26.05.17.
 */
window.onload = function () {
    var sampleTable = document.getElementById('sample');
    var sampleHead = sampleTable.querySelector('thead');
    var sampleBody = sampleTable.querySelector('tbody');
    var len = sample_chars.length;
    var nLayer = sample_states[0].length;
    var nCState = sample_states[0][0].c.length;
    var nHState = sample_states[0][0].h.length;
    function assert(pred, msg) {
        if (!pred)
            throw new Error(msg || "");
    }
    function repr(char) {
        var res = char;
        res = res.replace(/\n/g, '\\n');
        res = res.replace(/\t/g, '\\t');
        return res;
    }
    function sigmoid(x) {
        return 1 / (1 + Math.exp(x));
    }
    function tanh2Sigmoid(x) {
        return (x + 1) / 2;
    }
    function sigmoid2Color(x) {
        return Math.floor(x * 405.9);
    }
    function rgb(r, g, b) {
        return 'rgb(' + r.toString() + ',' + g.toString() + ',' + b.toString() + ')';
    }
    function renderSample() {
        assert(sample_chars.length === sample_states.length, "sample_chars.length === sample_states.length");
        // TODO: check the consistency of state shapes
        // render chars
        for (var i = 0; i < len; ++i) {
            var x = document.createElement('th');
            x.textContent = repr(sample_chars[i]);
            x.setAttribute('data-index', i.toString());
            if ('\n\t'.indexOf(sample_chars[i]) >= 0)
                x.classList.add('special-char');
            sampleHead.appendChild(x);
        }
        // render states
        for (var i = 0; i < nLayer; ++i) {
            for (var _i = 0, _a = ['c', 'h']; _i < _a.length; _i++) {
                var ch = _a[_i];
                for (var j = 0; j < (ch === 'c' ? nCState : nHState); ++j) {
                    var row = document.createElement('tr');
                    var vals = [];
                    for (var k = 0; k < len; ++k) {
                        var cell = document.createElement('td');
                        var val = sample_states[k][i][ch][j];
                        vals.push(val);
                        val = (ch === 'c' ? sigmoid(val) : tanh2Sigmoid(val));
                        val = sigmoid2Color(val);
                        var color = (ch === 'c' ? rgb(val / 2, val / 2, val) : rgb(val, val / 2, val / 2));
                        cell.style.backgroundColor = color;
                        row.appendChild(cell);
                    }
                    row.setAttribute('data-ch', ch);
                    row.setAttribute('data-layer', i.toString());
                    row.setAttribute('data-state-index', j.toString());
                    row.setAttribute('data-absdiff', calcAbsDiff(vals).toString());
                    row.setAttribute('data-signchanges', calcSignChanges(vals).toString());
                    sampleBody.appendChild(row);
                }
            }
        }
    }
    function setupScroll() {
        var table = document.querySelector('#sample');
        window.onscroll = function () {
            sampleHead.style.top = Math.max(0, -table.getBoundingClientRect().top) + 'px';
        };
    }
    function setupSelection() {
        var timerId = null;
        var selectListener = function (ev) {
            console.log('selecting...');
            var th = ev.currentTarget;
            th.classList.toggle('char-selected');
            console.log('selected.');
            // if (timerId)
            //     clearTimeout(timerId);
            // timerId = setTimeout(
            //     () => {
            //         sampleHead.classList.add('sorting');
            //         setTimeout(
            //             () => {
            //                 sortStates();
            //                 sampleHead.classList.remove('sorting');
            //             },
            //             50);
            //     },
            //     1000
            // );
        };
        var ths = document.querySelectorAll('th');
        for (var i = 0; i < ths.length; ++i)
            ths[i].onclick = selectListener;
    }
    function sortStates() {
        console.log('sorting...');
        var ths = sampleHead.querySelectorAll('th.char-selected');
        var selected = [];
        for (var i = 0; i < ths.length; ++i)
            selected.push(parseInt(ths[i].getAttribute('data-index')));
        var sortdata = [];
        var rows = document.querySelectorAll('tbody>tr');
        for (var i = 0; i < rows.length; ++i) {
            var row = rows[i];
            var ch = row.getAttribute('data-ch');
            var layer = parseInt(row.getAttribute('data-layer'));
            var stateIdx = parseInt(row.getAttribute('data-state-index'));
            var selvals = [];
            for (var _i = 0, selected_1 = selected; _i < selected_1.length; _i++) {
                var j = selected_1[_i];
                selvals.push(sample_states[j][layer][ch][stateIdx]);
            }
            // const variance = parseFloat(row.getAttribute('data-absdiff'));
            // const selvariance = calcVariance(selvals);
            // const key = selvariance / (variance+1e-3);
            // const absDiff = parseFloat(row.getAttribute('data-absdiff'));
            // const selAbsDiff = calcAbsDiff(selvals);
            // const key = selAbsDiff / (absDiff + 1e-3);
            var key = parseFloat(row.getAttribute('data-signchanges'));
            sortdata.push({ key: key, row: row });
        }
        sortdata.sort(function (a, b) { return a.key - b.key; });
        while (sampleBody.lastChild)
            sampleBody.removeChild(sampleBody.lastChild);
        for (var _a = 0, sortdata_1 = sortdata; _a < sortdata_1.length; _a++) {
            var x = sortdata_1[_a];
            sampleBody.appendChild(x.row);
        }
        console.log('sorted.');
    }
    function indexed(a, idx) {
        var res = [];
        for (var _i = 0, idx_1 = idx; _i < idx_1.length; _i++) {
            var i = idx_1[_i];
            res.push(a[i]);
        }
        return res;
    }
    function calcMean(a) {
        var len = a.length;
        var sum = 0;
        for (var _i = 0, a_1 = a; _i < a_1.length; _i++) {
            var ai = a_1[_i];
            sum += ai;
        }
        return sum / len;
    }
    function calcVariance(a) {
        var len = a.length;
        var m = calcMean(a);
        var sumsq = 0;
        for (var _i = 0, a_2 = a; _i < a_2.length; _i++) {
            var ai = a_2[_i];
            sumsq += (ai - m) * (ai - m);
        }
        return sumsq / len;
    }
    function calcAbsDiff(a) {
        var sum = 0;
        for (var i = 1; i < a.length; ++i)
            sum += Math.pow(Math.abs(a[i - 1] - a[i]), 2);
        return sum;
    }
    function calcSignChanges(a) {
        var cnt = 0;
        for (var i = 2; i < a.length; ++i)
            if ((a[i - 2] - a[i - 1]) * (a[i - 1] - a[i]) < 0)
                cnt += 1;
        return cnt;
    }
    renderSample();
    setupScroll();
    setupSelection();
    document.querySelector('#sort').onclick = function () {
        sortStates();
    };
};
