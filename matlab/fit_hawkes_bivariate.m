function results = fit_hawkes_bivariate()

clc;

fprintf('MATLAB BIVARIATE HAWKES EXACT-GRID REPLICATION\n');
fprintf('==============================================\n\n');

captureFiles = {
    'data/live/capture_02.jsonl'
    'data/live/capture_03.jsonl'
    'data/live/capture_04.jsonl'
};

captureNames = {
    'capture_02'
    'capture_03'
    'capture_04'
};

bookFile = 'data/processed/all_capture_book_states.parquet';

dt = 0.1;

fprintf('Loading reconstructed book states...\n');

book = parquetread(bookFile);

fprintf('Book rows loaded: %d\n\n', height(book));

results = table();

for k = 1:length(captureFiles)

    captureName = captureNames{k};

    fprintf('Processing %s...\n', captureName);

    tradeTimes = readTradeTimes(captureFiles{k});

    mask = strcmp(string(book.capture_id), captureName);

    localBook = book(mask, :);

    if height(localBook) == 0
        error('No book states found for %s.', captureName);
    end

    bookTimes = double(localBook.event_time_ms);

    startMs = min([min(tradeTimes), min(bookTimes)]);

    tradeBins = floor((double(tradeTimes) - startMs) / (dt * 1000));

    bookBins = floor((bookTimes - startMs) / (dt * 1000));

    maximumBin = max([max(tradeBins), max(bookBins)]);

    nBins = maximumBin + 1;

    buyCounts = zeros(nBins, 1);
    sellCounts = zeros(nBins, 1);

    sellFlags = readTradeSides(captureFiles{k});

    if length(sellFlags) ~= length(tradeBins)
        error('Trade timestamp/side count mismatch in %s.', captureName);
    end

    for i = 1:length(tradeBins)

        b = tradeBins(i) + 1;

        if sellFlags(i)
            sellCounts(b) = sellCounts(b) + 1;
        else
            buyCounts(b) = buyCounts(b) + 1;
        end

    end

    totalTrades = sum(buyCounts) + sum(sellCounts);

    if totalTrades ~= length(tradeTimes)
        error('Trade count changed during binning in %s.', captureName);
    end

    fprintf('Start ms: %.0f\n', startMs);
    fprintf('Book states: %d\n', height(localBook));
    fprintf('Bins: %d\n', nBins);
    fprintf('Buy events: %d\n', sum(buyCounts));
    fprintf('Sell events: %d\n', sum(sellCounts));
    fprintf('Total trades: %d\n', totalTrades);

    p = estimateHawkes(buyCounts, sellCounts, dt);

    spectralRadius = max(p.nBuy, p.nSell);

    fprintf('mu_buy: %.8f\n', p.muBuy);
    fprintf('mu_sell: %.8f\n', p.muSell);
    fprintf('beta: %.8f\n', p.beta);
    fprintf('branching_buy: %.8f\n', p.nBuy);
    fprintf('branching_sell: %.8f\n', p.nSell);
    fprintf('spectral_radius: %.8f\n', spectralRadius);
    fprintf('negative_log_likelihood: %.8f\n', p.negLogLikelihood);
    fprintf('\n');

    newRow = table( ...
        string(captureName), ...
        nBins, ...
        height(localBook), ...
        sum(buyCounts), ...
        sum(sellCounts), ...
        p.muBuy, ...
        p.muSell, ...
        p.beta, ...
        p.nBuy, ...
        p.nSell, ...
        spectralRadius, ...
        p.negLogLikelihood, ...
        'VariableNames', { ...
        'capture_id', ...
        'bins', ...
        'book_states', ...
        'buy_events', ...
        'sell_events', ...
        'mu_buy', ...
        'mu_sell', ...
        'beta', ...
        'branching_buy', ...
        'branching_sell', ...
        'spectral_radius', ...
        'negative_log_likelihood'});

    results = [results; newRow];

end

fprintf('========================================\n');
fprintf('FINAL MATLAB RESULTS\n');
fprintf('========================================\n\n');

disp(results);

outputFile = 'data/processed/matlab_hawkes_exact_replication.csv';

writetable(results, outputFile);

fprintf('\nSaved to: %s\n', outputFile);

end


function tradeTimes = readTradeTimes(filename)

fid = fopen(filename, 'r');

if fid == -1
    error('Could not open %s.', filename);
end

tradeTimes = [];

while true

    line = fgetl(fid);

    if ~ischar(line)
        break;
    end

    line = strtrim(line);

    if isempty(line)
        continue;
    end

    record = jsondecode(line);

    if ~isfield(record, 'type')
        continue;
    end

    if ~strcmp(record.type, 'trade')
        continue;
    end

    event = record.data;

    if ~isfield(event, 'T')
        fclose(fid);
        error('Trade record missing T.');
    end

    tradeTimes(end + 1, 1) = double(event.T);

end

fclose(fid);

if isempty(tradeTimes)
    error('No trades found in %s.', filename);
end

end


function sellFlags = readTradeSides(filename)

fid = fopen(filename, 'r');

if fid == -1
    error('Could not open %s.', filename);
end

sellFlags = [];

while true

    line = fgetl(fid);

    if ~ischar(line)
        break;
    end

    line = strtrim(line);

    if isempty(line)
        continue;
    end

    record = jsondecode(line);

    if ~isfield(record, 'type')
        continue;
    end

    if ~strcmp(record.type, 'trade')
        continue;
    end

    event = record.data;

    if ~isfield(event, 'm')
        fclose(fid);
        error('Trade record missing m.');
    end

    sellFlags(end + 1, 1) = logical(event.m);

end

fclose(fid);

if isempty(sellFlags)
    error('No trade sides found in %s.', filename);
end

end


function p = estimateHawkes(buy, sell, dt)

buy = double(buy(:));
sell = double(sell(:));

buyRate = mean(buy) / dt;
sellRate = mean(sell) / dt;

branchingLimit = 0.999;

betaStarts = [1 3 5 10];
branchingStarts = [0.20 0.40 0.60 0.75];

bestValue = Inf;
bestX = [];

options = optimset( ...
    'Display', 'off', ...
    'MaxIter', 1500, ...
    'MaxFunEvals', 10000, ...
    'TolX', 1e-8, ...
    'TolFun', 1e-10);

for i = 1:length(betaStarts)

    for j = 1:length(branchingStarts)

        n0 = branchingStarts(j);

        z = log(n0 / (branchingLimit - n0));

        x0 = zeros(5, 1);

        x0(1) = log(max(buyRate * 0.5, 1e-6));
        x0(2) = log(max(sellRate * 0.5, 1e-6));
        x0(3) = log(betaStarts(i));
        x0(4) = z;
        x0(5) = z;

        objective = @(x) hawkesNLL( ...
            x, ...
            buy, ...
            sell, ...
            dt, ...
            branchingLimit);

        try

            [x, fval, exitflag] = fminsearch( ...
                objective, ...
                x0, ...
                options);

        catch

            continue;

        end

        if exitflag > 0

            if isfinite(fval)

                if fval < bestValue
                    bestValue = fval;
                    bestX = x;
                end

            end

        end

    end

end

if isempty(bestX)
    error('Hawkes optimization failed.');
end

[muBuy, muSell, beta, nBuy, nSell] = ...
    decodeParameters(bestX, branchingLimit);

p.muBuy = muBuy;
p.muSell = muSell;
p.beta = beta;
p.nBuy = nBuy;
p.nSell = nSell;
p.negLogLikelihood = bestValue;

end


function value = hawkesNLL(x, buy, sell, dt, branchingLimit)

[muBuy, muSell, beta, nBuy, nSell] = ...
    decodeParameters(x, branchingLimit);

decay = exp(-beta * dt);

scale = 1 - decay;

buyState = zeros(size(buy));
sellState = zeros(size(sell));

if length(buy) > 1

    buyState(2:end) = filter( ...
        1, ...
        [1 -decay], ...
        buy(1:end-1));

    sellState(2:end) = filter( ...
        1, ...
        [1 -decay], ...
        sell(1:end-1));

end

meanBuy = muBuy * dt + ...
    nBuy * scale .* buyState;

meanSell = muSell * dt + ...
    nSell * scale .* sellState;

if any(~isfinite(meanBuy))
    value = 1e100;
    return;
end

if any(~isfinite(meanSell))
    value = 1e100;
    return;
end

if any(meanBuy <= 0)
    value = 1e100;
    return;
end

if any(meanSell <= 0)
    value = 1e100;
    return;
end

logLikelihoodBuy = ...
    buy .* log(meanBuy) ...
    - meanBuy ...
    - gammaln(buy + 1);

logLikelihoodSell = ...
    sell .* log(meanSell) ...
    - meanSell ...
    - gammaln(sell + 1);

logLikelihood = ...
    sum(logLikelihoodBuy) ...
    + sum(logLikelihoodSell);

value = -logLikelihood;

if ~isfinite(value)
    value = 1e100;
end

end


function [muBuy, muSell, beta, nBuy, nSell] = ...
    decodeParameters(x, branchingLimit)

muBuy = exp(min(max(x(1), -20), 20));

muSell = exp(min(max(x(2), -20), 20));

beta = exp(min(max(x(3), -10), 10));

nBuy = branchingLimit * sigmoid(x(4));

nSell = branchingLimit * sigmoid(x(5));

end


function y = sigmoid(x)

if x >= 0

    y = 1 / (1 + exp(-x));

else

    e = exp(x);

    y = e / (1 + e);

end

end