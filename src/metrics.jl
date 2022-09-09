module Metrics


export acc, precision, recall, specificity, f1, confusionmatrix


function acc(y, ŷ)
    p, n, tp, tn, fp, fn = cmmetrics(y, ŷ)
    (tp + tn) / (p + n)
end

function precision(y, ŷ)
    p, n, tp, tn, fp, fn = cmmetrics(y, ŷ)
    tp / p
end

function recall(y, ŷ)
    p, n, tp, tn, fp, fn = cmmetrics(y, ŷ)
    tp / p
end

function specificity(y, ŷ)
    p, n, tp, tn, fp, fn = cmmetrics(y, ŷ)
    tn / n
end

function f1(y, ŷ)
    prec = precision(y, ŷ)
    rec = recall(y, ŷ)
    2 * (prec * rec) / (prec + rec)
end

function cmmetrics(y, ŷ)
    t = length(y)
    p = length(y[y .== 1])
    n = t - p

    tp = fp = tn = fn = 0

    for (yᵢ, ŷᵢ) ∈ zip(y, ŷ)
        if (yᵢ == ŷᵢ)
            if (yᵢ == 0)
                tn += 1
            else
                tp += 1
            end
        else
            if (yᵢ == 0)
                fp += 1
            else
                fn += 1
            end
        end
    end

    p, n, tp, tn, fp, fn
end

function confusionmatrix(y, ŷ)
    p, n, tp, tn, fp, fn = cmmetrics(y, ŷ)
    accuracy = (tp + tn) / (p + n)
    prec = tp / (tp + fp)
    rec = tp / p
    spec = tn / n
    f1 = 2 * (prec * rec) / (prec + rec)

    cm = [
        tp fp
        fn tn
    ]
    
    println("Confusion Matrix Metrics")
    println("---")
    println("Accuracy: $accuracy")
    println("Precision: $prec")
    println("Recall: $rec")
    println("Specificity: $spec")
    println("F1: $f1")
    println("---")
    println(cm)
    
end

end