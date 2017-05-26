/**
 * Created by exprosic on 26.05.17.
 */

declare const sample_chars: Array<string>;
declare const sample_states: Array<Array<{c: Array<number>, h: Array<number>}>>;

window.onload = () => {
    const sampleTable = document.getElementById('sample') as HTMLTableElement;
    const sampleHead = sampleTable.querySelector('thead') as HTMLTableSectionElement;
    const sampleBody = sampleTable.querySelector('tbody') as HTMLTableSectionElement;

    const len = sample_chars.length;
    const nLayer = sample_states[0].length;
    const nCState = sample_states[0][0].c.length;
    const nHState = sample_states[0][0].h.length;

    function assert(pred: boolean, msg?: string): void {
        if (!pred)
            throw new Error(msg || "");
    }

    function repr(char: string): string {
        let res = char;
        res = res.replace(/\n/g, '\\n');
        res = res.replace(/\t/g, '\\t');
        return res;
    }

    function sigmoid(x: number): number {
        return 1/(1+Math.exp(x));
    }

    function tanh2Sigmoid(x: number): number {
        return (x+1)/2;
    }

    function sigmoid2Color(x: number): number {
        return Math.floor(x * 405.9);
    }

    function rgb(r: number, g: number, b: number): string {
        return 'rgb(' + r.toString() + ',' + g.toString() + ',' + b.toString() + ')';
    }

    function renderSample(): void {
        assert(sample_chars.length === sample_states.length, "sample_chars.length === sample_states.length");
        // TODO: check the consistency of state shapes

        // render chars
        for (let i=0; i<len; ++i) {
            const x = document.createElement('th');
            x.textContent = repr(sample_chars[i]);
            x.setAttribute('data-index', i.toString());
            if ('\n\t'.indexOf(sample_chars[i]) >= 0)
                x.classList.add('special_char');
            sampleHead.appendChild(x);
        }

        // render states
        for (let i=0; i<nLayer; ++i) {
            for (let ch of ['c','h']) {
                for (let j=0; j<(ch==='c' ?nCState :nHState); ++j) {
                    const row = document.createElement('tr');
                    const vals = [];
                    for (let k=0; k<len; ++k) {
                        const cell = document.createElement('td');

                        let val = sample_states[k][i][ch][j];
                        vals.push(val);
                        val = (ch==='c' ?sigmoid(val) :tanh2Sigmoid(val));
                        val = sigmoid2Color(val);
                        const color = (ch==='c' ?rgb(val/2, val/2, val) :rgb(val, val/2, val/2));

                        cell.style.backgroundColor = color;
                        row.appendChild(cell);
                    }
                    row.setAttribute('data-ch', ch);
                    row.setAttribute('data-layer', i.toString());
                    row.setAttribute('data-state-index', j.toString());
                    row.setAttribute('data-variance', calcVariance(vals).toString());
                    sampleBody.appendChild(row);
                }
            }
        }
    }

    function setupScroll(): void {
        const table = document.querySelector('#sample');
        window.onscroll = () => {
            sampleHead.style.top = Math.max(0, -table.getBoundingClientRect().top) + 'px';
        };
    }

    function setupSelection(): void {
        const selectListener = (ev) => {
            console.log('selecting...');
            const th = ev.currentTarget as HTMLTableHeaderCellElement;
            if (!th.hasAttribute('char-selected')) {
                th.setAttribute('char-selected', "");
            } else {
                th.removeAttribute('char-selected');
            }
            console.log('selected.');

            sortStates();
        };
        const ths = document.querySelectorAll('th');
        for (let i=0; i<ths.length; ++i)
            ths[i].onclick = selectListener;

        const variances: Array<number> = [];

        function sortStates(): void {
            console.log('sorting...');
            const ths = sampleHead.querySelectorAll('th[char-selected]');
            let selected: Array<number> = [];
            for (let i=0; i<ths.length; ++i)
                selected.push(parseInt(ths[i].getAttribute('data-index')));

            const sortdata: Array<{key: number, row: HTMLTableRowElement}> = [];
            const rows = document.querySelectorAll('tbody>tr');
            for (let i=0; i<rows.length; ++i) {
                const row = rows[i] as HTMLTableRowElement;
                const ch = row.getAttribute('data-ch') as ('c'|'h');
                const layer = parseInt(row.getAttribute('data-layer'));
                const stateIdx = parseInt(row.getAttribute('data-state-index'));
                const selvals: Array<number> = [];
                for (let j of selected)
                    selvals.push(sample_states[j][layer][ch][stateIdx]);
                const variance = parseFloat(row.getAttribute('data-variance'));
                const selvariance = calcVariance(selvals);
                const key = selvariance / (variance+1e-3);
                sortdata.push({key: key, row: row});
            }

            sortdata.sort((a,b) => a.key-b.key);
            while (sampleBody.lastChild)
                sampleBody.removeChild(sampleBody.lastChild);
            for (let x of sortdata)
                sampleBody.appendChild(x.row);
            console.log('sorted.');
        }
    }

    function indexed<T>(a: Array<T>, idx: Array<number>): Array<T> {
        const res: Array<T> = [];
        for (let i of idx)
            res.push(a[i]);
        return res;
    }

    function calcMean(a: Array<number>): number {
        const len = a.length;
        let sum = 0;
        for (let ai of a)
            sum += ai;
        return sum/len;
    }

    function calcVariance(a: Array<number>): number {
        const len = a.length;
        const m = calcMean(a);
        let sumsq = 0;
        for (let ai of a)
            sumsq += (ai-m) * (ai-m);
        return sumsq/len;
    }

    renderSample();
    setupScroll();
    setupSelection();
};