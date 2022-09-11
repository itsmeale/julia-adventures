module Metrics


export acc, precision, recall, specificity, f1, classification_report, multiclass_report


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

function classification_report(y, ŷ)
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
    println("Instances: $(p + n) ($p positives, $n negatives)")
    println("True Positive: $tp")
    println("True Negative: $tn")
    println("False Positive: $fp")
    println("False Negative: $fn")
    println("Accuracy: $accuracy")
    println("Precision: $prec")
    println("Recall: $rec")
    println("Specificity: $spec")
    println("F1: $f1")
    println("---")
end

function multiclass_report(Y, Ŷ)
    k = size(Y)[2]
    predicted_classes = getclasses(Ŷ)
    Ŷₛ = signal(Ŷ)

    global_acc = acc(Y, predicted_classes)
    println("Overall accuracy: $global_acc")
    for i ∈ 1:k
        println("\nMetrics for class $i")
        classification_report(Y[:, i], Ŷₛ[:, i])
    end
end

#= Obtem o indice do maior valor de y para cada instancia  =#
function getclasses(Ŷ)
    return mapslices(argmax, Ŷ, dims=2)[:]
end

#= Mapeia a classe mais provavel para 1 e as outras para 0 =#
function signal(Y)
    Yc = copy(Y)
    n, k = size(Yc)
    classes = getclasses(Yc)
    for j in 1:k
        for i in 1:n
            j == classes[i] ? (Yc[i, j] = 1) : (Yc[i, j] = 0)
        end
    end
    return Yc
end

end